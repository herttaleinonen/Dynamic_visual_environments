# Created on Mon Feb 26 11:38:18 2026
# @author: herttaleinonen

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)
library(car)
library(performance)
library(broom.mixed)
library(moments)  
library(patchwork)

# ============================================================
# Load data
# ============================================================

dat <- read_csv("data/long.csv", show_col_types = FALSE)

# ============================================================
# Speed mapping
# ============================================================

speed_map <- tibble(
  task      = paste0("dt", 1:5),
  speed_num = c(0.000, 2.703, 5.406, 8.109, 10.812),
  speed_lab = c("0", "3", "5.5", "8", "11")
)

speed_breaks <- speed_map$speed_num
speed_labels <- speed_map$speed_lab

# ============================================================
# DT tasks + factors
# ============================================================

dt <- dat %>%
  filter(task %in% paste0("dt", 1:5)) %>%
  left_join(speed_map, by = "task") %>%
  mutate(
    speed_fac      = factor(speed_lab, levels = speed_labels),
    participant    = factor(participant),
    target_present = factor(target_present, levels = c(0, 1),
                            labels = c("absent", "present"))
  )

# -0.5 / +0.5 contrast coding: coefficient = direct condition difference
# (vs contr.sum which gives difference/2)
contrasts(dt$target_present) <- c(-0.5, 0.5)

# ============================================================
# RT DISTRIBUTION DIAGNOSTICS
# ============================================================

rt_raw <- dt %>% filter(correct == 1, !is.na(rt_s))

cat("\n=== RT descriptives ===\n")
cat(sprintf("Mean:     %.3f s\n", mean(rt_raw$rt_s)))
cat(sprintf("Median:   %.3f s\n", median(rt_raw$rt_s)))
cat(sprintf("SD:       %.3f s\n", sd(rt_raw$rt_s)))
cat(sprintf("Skewness: %.3f\n",   skewness(rt_raw$rt_s)))

set.seed(42)
rt_sample <- sample(rt_raw$rt_s, min(5000, nrow(rt_raw)))

sw_raw <- shapiro.test(rt_sample)
ks_raw <- ks.test(scale(rt_sample), "pnorm")
sw_log <- shapiro.test(log(rt_sample))
ks_log <- ks.test(scale(log(rt_sample)), "pnorm")

cat("\n=== Normality tests: raw RT ===\n")
cat(sprintf("Shapiro-Wilk:       W = %.4f, p = %.4e\n", sw_raw$statistic, sw_raw$p.value))
cat(sprintf("Kolmogorov-Smirnov: D = %.4f, p = %.4e\n", ks_raw$statistic, ks_raw$p.value))

cat("\n=== Normality tests: log(RT) ===\n")
cat(sprintf("Shapiro-Wilk:       W = %.4f, p = %.4e\n", sw_log$statistic, sw_log$p.value))
cat(sprintf("Kolmogorov-Smirnov: D = %.4f, p = %.4e\n", ks_log$statistic, ks_log$p.value))

p_hist_raw <- ggplot(rt_raw, aes(x = rt_s)) +
  geom_histogram(bins = 60, fill = "#1f77b4", alpha = 0.7) +
  labs(title = "Raw RT distribution", x = "RT (s)", y = "Count") +
  theme_minimal(base_size = 14)

p_hist_log <- ggplot(rt_raw, aes(x = log(rt_s))) +
  geom_histogram(bins = 60, fill = "#d62728", alpha = 0.7) +
  labs(title = "Log RT distribution", x = "log(RT)", y = "Count") +
  theme_minimal(base_size = 14)

p_qq_raw <- ggplot(rt_raw, aes(sample = rt_s)) +
  stat_qq() + stat_qq_line(color = "red") +
  labs(title = "Q-Q plot: raw RT") +
  theme_minimal(base_size = 14)

p_qq_log <- ggplot(rt_raw, aes(sample = log(rt_s))) +
  stat_qq() + stat_qq_line(color = "red") +
  labs(title = "Q-Q plot: log(RT)") +
  theme_minimal(base_size = 14)

# Save RT distribution diagnostics to PDF
pdf("diagnostics_rt_distribution.pdf", width = 10, height = 8)
print(p_hist_raw)
print(p_hist_log)
print(p_qq_raw)
print(p_qq_log)
dev.off()
cat("RT distribution diagnostics saved to diagnostics_rt_distribution.pdf\n")

# ============================================================
# Aggregate data
# ============================================================

dt_rt <- dt %>%
  filter(correct == 1) %>%
  group_by(participant, speed_num, speed_fac, target_present) %>%
  summarise(
    rt     = mean(rt_s,      na.rm = TRUE),
    log_rt = mean(log(rt_s), na.rm = TRUE),
    .groups = "drop"
  )

# Accuracy: logit-transformed averaged proportions
n_trials_per_cell <- dt %>%
  group_by(participant, speed_num, target_present) %>%
  summarise(n = n(), .groups = "drop") %>%
  pull(n) %>%
  median()

eps <- 0.5 / n_trials_per_cell
cat(sprintf("\nAccuracy continuity correction eps = %.4f (based on median %d trials/cell)\n",
            eps, round(n_trials_per_cell)))

dt_acc <- dt %>%
  group_by(participant, speed_num, speed_fac, target_present) %>%
  summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    acc_clipped = pmin(acc, 1 - eps),
    logit_acc   = log(acc_clipped / (1 - acc_clipped))
  )

dt_eye <- dt %>%
  group_by(participant, speed_num, speed_fac, target_present) %>%
  summarise(
    fix_count   = mean(fix_count,           na.rm = TRUE),
    scanpath    = mean(fix_path_length_deg, na.rm = TRUE),
    dispersion  = mean(fix_dispersion_deg2, na.rm = TRUE),
    center_dist = mean(fix_center_dist_deg, na.rm = TRUE),
    n_trials    = sum(!is.na(fix_center_dist_deg)),
    .groups = "drop"
  ) %>%
  filter(n_trials > 0)

# ============================================================
# Mixed models — random slopes
#
# Full model:  (1 + speed_num + target_present | participant)
#   allows each participant to have their own velocity slope
#   and their own absent/present difference
#
# Fallback if convergence fails:
#   (1 + speed_num || participant)   — removes random effect correlations
# ============================================================

fit_lmm_slopes <- function(formula_str, data) {
  m_full <- tryCatch(
    lmer(as.formula(formula_str), data = data, REML = FALSE),
    warning = function(w) {
      cat(sprintf("[WARN] Full random slopes model: %s\nFalling back to uncorrelated slopes.\n",
                  conditionMessage(w)))
      NULL
    }
  )
  if (!is.null(m_full)) return(m_full)

  # Fallback: uncorrelated random slopes (speed only)
  formula_fallback <- gsub(
    "\\(1 \\+ speed_num \\+ target_present \\| participant\\)",
    "(1 + speed_num || participant)",
    formula_str
  )
  lmer(as.formula(formula_fallback), data = data, REML = FALSE)
}

# ------ RT: log-transformed (chosen based on AIC and residual diagnostics) ------
m_rt_log <- fit_lmm_slopes(
  "log_rt ~ speed_num * target_present + (1 + speed_num + target_present | participant)",
  dt_rt
)

# Keep raw RT model for AIC comparison only
m_rt_raw <- lmer(
  rt ~ speed_num * target_present + (1 + speed_num + target_present | participant),
  data = dt_rt, REML = FALSE
)

cat("\n=== RT model comparison (AIC, random slopes) ===\n")
cat(sprintf("Raw RT LMM:   AIC = %.1f\n", AIC(m_rt_raw)))
cat(sprintf("Log RT LMM:   AIC = %.1f\n", AIC(m_rt_log)))

# Final RT model
m_rt <- m_rt_log

# ------ Accuracy: logit LMM with random slopes ------
m_acc <- fit_lmm_slopes(
  "logit_acc ~ speed_num * target_present + (1 + speed_num + target_present | participant)",
  dt_acc
)

# ------ Eye movement models with random slopes ------
m_fix  <- fit_lmm_slopes(
  "fix_count ~ speed_num * target_present + (1 + speed_num + target_present | participant)",
  dt_eye
)
m_scan <- fit_lmm_slopes(
  "scanpath ~ speed_num * target_present + (1 + speed_num + target_present | participant)",
  dt_eye
)
m_disp <- fit_lmm_slopes(
  "dispersion ~ speed_num * target_present + (1 + speed_num + target_present | participant)",
  dt_eye
)
m_ctr <- fit_lmm_slopes(
  "center_dist ~ speed_num * target_present + (1 + speed_num + target_present | participant)",
  dt_eye
)

# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

check_residuals <- function(model, label) {
  res <- residuals(model)
  fit <- fitted(model)

  p1 <- ggplot(data.frame(res = res), aes(sample = res)) +
    stat_qq() + stat_qq_line(color = "red") +
    labs(title = paste("Q-Q residuals:", label)) +
    theme_minimal(base_size = 14)

  p2 <- ggplot(data.frame(fit = fit, res = res), aes(x = fit, y = res)) +
    geom_point(alpha = 0.3, size = 0.8) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title = paste("Residuals vs fitted:", label), x = "Fitted", y = "Residuals") +
    theme_minimal(base_size = 14)

  print(p1)
  print(p2)

  sw <- shapiro.test(sample(res, min(5000, length(res))))
  cat(sprintf("Residual Shapiro-Wilk [%s]: W = %.4f, p = %.4e\n",
              label, sw$statistic, sw$p.value))
}

cat("\n=== RESIDUAL DIAGNOSTICS ===\n")
pdf("diagnostics_residuals.pdf", width = 10, height = 5)
check_residuals(m_rt,   "RT log LMM")
check_residuals(m_acc,  "Accuracy logit LMM")
check_residuals(m_fix,  "Fixation count")
check_residuals(m_scan, "Scanpath length")
check_residuals(m_disp, "Dispersion")
check_residuals(m_ctr,  "Centre distance")
dev.off()
cat("Residual diagnostics saved to diagnostics_residuals.pdf\n")

# ============================================================
# ============================================================
# Colors + journal-compliant theme
# ============================================================

COND_COLORS <- c(
  "absent"  = "#d62728",
  "present" = "#1f77b4"
)

# Journal guidelines:
# - Helvetica font
# - 10pt axis numbers, 12pt axis labels
# - No bold
# - Axis lines and ticks
# - Minimal padding
journal_theme <- theme_minimal(base_size = 22, base_family = "Helvetica") +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor    = element_blank(),
    axis.text           = element_text(size = 18, face = "plain", family = "Helvetica"),
    axis.title          = element_text(size = 22, face = "plain", family = "Helvetica"),
    legend.text         = element_text(size = 18, face = "plain", family = "Helvetica"),
    legend.title        = element_text(size = 18, face = "plain", family = "Helvetica"),
    plot.tag            = element_text(size = 28, face = "plain", family = "Helvetica"),
    axis.ticks          = element_line(color = "#333333"),
    axis.ticks.length   = unit(3, "pt"),
    axis.line           = element_line(color = "#333333", linewidth = 0.5),
    plot.margin         = margin(6, 8, 6, 6)
  )

plot_spaghetti_with_lmm <- function(df, dv_name, model, ylab,
                                    speed_var    = "speed_num",
                                    group_var    = "participant",
                                    cond_var     = "target_present",
                                    speed_breaks,
                                    speed_labels,
                                    show_x_label = TRUE,
                                    show_legend  = TRUE) {

  df <- df %>% filter(!is.na(.data[[cond_var]]))

  means <- df %>%
    group_by(.data[[speed_var]], .data[[cond_var]]) %>%
    summarise(
      mean = mean(.data[[dv_name]], na.rm = TRUE),
      sd   = sd(.data[[dv_name]],   na.rm = TRUE),
      n    = dplyr::n(),
      .groups = "drop"
    ) %>%
    mutate(se = sd / sqrt(n), ci = qt(0.975, df = n - 1) * se) %>%
    rename(speed = .data[[speed_var]], cond = .data[[cond_var]])

  pred_grid <- expand.grid(
    speed_num      = speed_breaks,
    target_present = levels(droplevels(df[[cond_var]]))
  )
  pred_grid$pred <- predict(model, newdata = pred_grid, re.form = NA)
  names(pred_grid)[names(pred_grid) == "speed_num"]      <- speed_var
  names(pred_grid)[names(pred_grid) == "target_present"] <- cond_var

  x_label <- if (show_x_label) "Velocity (deg/s)" else NULL

  p <- ggplot(df, aes(x = .data[[speed_var]], y = .data[[dv_name]],
                 color = .data[[cond_var]])) +
    geom_line(aes(group = interaction(.data[[group_var]], .data[[cond_var]])),
              alpha = 0.10, linewidth = 0.5) +
    geom_errorbar(data = means, inherit.aes = FALSE,
                  aes(x = speed, ymin = mean - ci, ymax = mean + ci, color = cond),
                  width = 0.25, linewidth = 0.8) +
    geom_point(data = means, inherit.aes = FALSE,
               aes(x = speed, y = mean, color = cond), size = 2.8) +
    geom_line(data = pred_grid,
              aes(x = .data[[speed_var]], y = pred,
                  color = .data[[cond_var]], group = .data[[cond_var]]),
              linewidth = 1.4) +
    scale_color_manual(values = COND_COLORS, na.translate = FALSE) +
    scale_x_continuous(breaks = speed_breaks, labels = speed_labels) +
    labs(x = x_label, y = ylab, color = "Target") +
    journal_theme

  if (!show_legend) p <- p + theme(legend.position = "none")
  p
}

# ============================================================
# Generate plots
# ============================================================

# --- Figure 5: RT (a) and Accuracy (b) ---

p_rt_spag <- ggplot(dt_rt, aes(x = speed_num, y = rt, color = target_present)) +
  geom_line(aes(group = interaction(participant, target_present)),
            alpha = 0.10, linewidth = 0.5) +
  geom_errorbar(
    data = dt_rt %>%
      group_by(speed_num, target_present) %>%
      summarise(mean = mean(rt, na.rm = TRUE),
                sd   = sd(rt,   na.rm = TRUE),
                n    = n(), .groups = "drop") %>%
      mutate(se = sd / sqrt(n), ci = qt(0.975, df = n - 1) * se),
    inherit.aes = FALSE,
    aes(x = speed_num, ymin = mean - ci, ymax = mean + ci, color = target_present),
    width = 0.25, linewidth = 0.8
  ) +
  geom_point(
    data = dt_rt %>%
      group_by(speed_num, target_present) %>%
      summarise(mean = mean(rt, na.rm = TRUE), .groups = "drop"),
    inherit.aes = FALSE,
    aes(x = speed_num, y = mean, color = target_present), size = 2.8
  ) +
  geom_line(
    data = {
      pg <- expand.grid(speed_num      = speed_breaks,
                        target_present = levels(dt_rt$target_present))
      pg$pred <- exp(predict(m_rt, newdata = pg, re.form = NA))
      pg
    },
    aes(x = speed_num, y = pred, color = target_present, group = target_present),
    linewidth = 1.4
  ) +
  scale_color_manual(values = COND_COLORS, na.translate = FALSE) +
  scale_x_continuous(breaks = speed_breaks, labels = speed_labels) +
  labs(x = "Velocity (deg/s)", y = "Reaction time (s)", color = "Target", tag = "a.") +
  journal_theme

pred_grid_acc <- expand.grid(
  speed_num      = speed_breaks,
  target_present = levels(dt_acc$target_present)
)
pred_grid_acc$pred_logit <- predict(m_acc, newdata = pred_grid_acc, re.form = NA)
pred_grid_acc$pred_acc   <- plogis(pred_grid_acc$pred_logit)

means_acc <- dt_acc %>%
  group_by(speed_num, target_present) %>%
  summarise(
    mean = mean(acc, na.rm = TRUE),
    sd   = sd(acc,   na.rm = TRUE),
    n    = n(),
    .groups = "drop"
  ) %>%
  mutate(se = sd / sqrt(n), ci = qt(0.975, df = n - 1) * se)

p_acc_spag <- ggplot(dt_acc, aes(x = speed_num, y = acc, color = target_present)) +
  geom_line(aes(group = interaction(participant, target_present)),
            alpha = 0.10, linewidth = 0.5) +
  geom_errorbar(data = means_acc, inherit.aes = FALSE,
                aes(x = speed_num, ymin = mean - ci, ymax = mean + ci,
                    color = target_present),
                width = 0.25, linewidth = 0.8) +
  geom_point(data = means_acc, inherit.aes = FALSE,
             aes(x = speed_num, y = mean, color = target_present), size = 2.8) +
  geom_line(data = pred_grid_acc,
            aes(x = speed_num, y = pred_acc,
                color = target_present, group = target_present),
            linewidth = 1.4) +
  scale_color_manual(values = COND_COLORS, na.translate = FALSE) +
  scale_x_continuous(breaks = speed_breaks, labels = speed_labels) +
  labs(x = "Velocity (deg/s)", y = "Accuracy", color = "Target", tag = "b.") +
  journal_theme

# --- Figure 6: Eye movement panels (a-d) ---
# All panels keep their legend data; patchwork collects into one shared legend
# x-axis label only on bottom panels

p_fix_spag <- plot_spaghetti_with_lmm(
  dt_eye, "fix_count", m_fix, "Fixation count",
  speed_breaks = speed_breaks, speed_labels = speed_labels,
  show_x_label = FALSE, show_legend = TRUE
) + labs(tag = "a.")

p_scan_spag <- plot_spaghetti_with_lmm(
  dt_eye, "scanpath", m_scan, "Scanpath length (deg)",
  speed_breaks = speed_breaks, speed_labels = speed_labels,
  show_x_label = FALSE, show_legend = TRUE
) + labs(tag = "b.")

p_disp_spag <- plot_spaghetti_with_lmm(
  dt_eye, "dispersion", m_disp, "Dispersion (deg\u00b2)",
  speed_breaks = speed_breaks, speed_labels = speed_labels,
  show_x_label = TRUE, show_legend = TRUE
) + labs(tag = "c.")

p_ctr_spag <- plot_spaghetti_with_lmm(
  dt_eye, "center_dist", m_ctr, "Distance from centre (deg)",
  speed_breaks = speed_breaks, speed_labels = speed_labels,
  show_x_label = TRUE, show_legend = TRUE
) + labs(tag = "d.")

# Print individual panels to PDF for inspection
pdf("figures_preview.pdf", width = 7, height = 5)
print(p_rt_spag)
print(p_acc_spag)
print(p_fix_spag)
print(p_scan_spag)
print(p_disp_spag)
print(p_ctr_spag)
dev.off()
cat("Individual panel previews saved to figures_preview.pdf\n")

# Assemble and print combined figures
fig5 <- p_rt_spag + p_acc_spag +
  plot_layout(ncol = 2, guides = "collect")
print(fig5)

fig6 <- (p_fix_spag | p_scan_spag) / (p_disp_spag | p_ctr_spag) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
print(fig6)

# Save as PDF (vector, for journal) and TIFF (for Word)
ggsave("figure5.pdf",  fig5, width = 14, height = 5,  device = cairo_pdf)
ggsave("figure5.tiff", fig5, width = 14, height = 5,  dpi = 300)

ggsave("figure6.pdf",  fig6, width = 14, height = 10, device = cairo_pdf)
ggsave("figure6.tiff", fig6, width = 14, height = 10, dpi = 300)

cat("\nFigures saved as PDF and EPS.\n")

# ============================================================
# Results tables and F-tests
# ============================================================

models <- list(
  "Reaction time (log s)" = m_rt,
  "Accuracy (logit)"      = m_acc,
  "Fixation count"        = m_fix,
  "Scanpath length (deg)" = m_scan,
  "Dispersion (deg2)"     = m_disp,
  "Centre distance (deg)" = m_ctr
)

# Beta coefficients table
results_table <- bind_rows(
  lapply(names(models), function(nm) {
    tidy(models[[nm]], effects = "fixed") %>%
      mutate(outcome = nm) %>%
      select(outcome, term, estimate, std.error, statistic, p.value)
  })
) %>%
  mutate(across(c(estimate, std.error, statistic), ~ round(., 3)),
         p.value = ifelse(p.value < .001, "< .001", as.character(round(p.value, 3))))

write.csv(results_table, "results_table.csv", row.names = FALSE)
print(results_table)

# ============================================================
# F-TESTS (lmerTest, Type III Satterthwaite)
# ============================================================

cat("\n=== F-TESTS (lmerTest anova, Satterthwaite) ===\n")

f_test_results <- lapply(names(models), function(nm) {
  cat(sprintf("\n--- %s ---\n", nm))
  ft <- anova(models[[nm]])
  print(ft)
  as.data.frame(ft) %>%
    mutate(outcome = nm, term = rownames(.)) %>%
    select(outcome, term, everything())
})

f_table <- bind_rows(f_test_results)
write.csv(f_table, "f_tests.csv", row.names = FALSE)
cat("\nF-tests written to f_tests.csv\n")

# ============================================================
# POST-HOC: velocity slopes separately for absent and present
# Use emtrends() when the velocity x target_present interaction
# is significant, to report velocity effects per condition
# rather than a single pooled coefficient.
# ============================================================

cat("\n=== VELOCITY SLOPES BY TARGET PRESENCE (emtrends) ===\n")

get_emtrends <- function(model, data, label) {
  cat(sprintf("\n--- %s ---\n", label))
  tryCatch({
    em <- emtrends(model, ~ target_present, var = "speed_num", data = data)
    print(summary(em))
    cat("Pairwise contrast (absent vs present velocity slope):\n")
    print(pairs(em))
    summary(em)
  }, error = function(e) {
    cat(sprintf("emtrends failed: %s\n", conditionMessage(e)))
    NULL
  })
}

em_rt   <- get_emtrends(m_rt,   dt_rt,  "Reaction time (log s)")
em_acc  <- get_emtrends(m_acc,  dt_acc, "Accuracy (logit)")
em_fix  <- get_emtrends(m_fix,  dt_eye, "Fixation count")
em_scan <- get_emtrends(m_scan, dt_eye, "Scanpath length")
em_disp <- get_emtrends(m_disp, dt_eye, "Dispersion")
em_ctr  <- get_emtrends(m_ctr,  dt_eye, "Centre distance")

# Save emtrends to CSV
em_list <- list(
  "Reaction time (log s)" = em_rt,
  "Accuracy (logit)"      = em_acc,
  "Fixation count"        = em_fix,
  "Scanpath length (deg)" = em_scan,
  "Dispersion (deg2)"     = em_disp,
  "Centre distance (deg)" = em_ctr
)

em_table <- bind_rows(
  lapply(names(em_list), function(nm) {
    if (!is.null(em_list[[nm]])) {
      as.data.frame(em_list[[nm]]) %>% mutate(outcome = nm)
    }
  })
)

write.csv(em_table, "emtrends_velocity_by_condition.csv", row.names = FALSE)
cat("\nVelocity slopes by condition written to emtrends_velocity_by_condition.csv\n")

