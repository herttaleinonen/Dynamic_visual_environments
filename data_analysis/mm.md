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

print(p_hist_raw)
print(p_hist_log)
print(p_qq_raw)
print(p_qq_log)

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
  summarise(
    acc = mean(correct, na.rm = TRUE),
    n_trials = sum(!is.na(correct)),   # track valid data
    .groups = "drop"
  ) %>%
  filter(
    n_trials > 0,                     # remove empty cells
    !is.na(target_present),           # remove NA condition
    is.finite(acc)                    # remove NaN
  ) %>%
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
#
# Try the full model first; if it warns about convergence,
# fall back to the simpler version
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
check_residuals(m_rt,   "RT log LMM")
check_residuals(m_acc,  "Accuracy logit LMM")
check_residuals(m_fix,  "Fixation count")
check_residuals(m_scan, "Scanpath length")
check_residuals(m_disp, "Dispersion")
check_residuals(m_ctr,  "Centre distance")

# ============================================================
# Colors + plotting function
# ============================================================

COND_COLORS <- c(
  "absent"  = "#d62728",
  "present" = "#1f77b4"
)

plot_spaghetti_with_lmm <- function(df, dv_name, model, ylab,
                                    speed_var    = "speed_num",
                                    group_var    = "participant",
                                    cond_var     = "target_present",
                                    speed_breaks,
                                    speed_labels) {

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

  ggplot(df, aes(x = .data[[speed_var]], y = .data[[dv_name]],
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
    labs(x = "Velocity (deg/s)", y = ylab, color = "Target") +
    theme_minimal(base_size = 18) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor    = element_blank(),
      plot.tag            = element_text(size = 24, face = "bold")
    )
}

# ============================================================
# Generate plots
# ============================================================

p_rt_spag <- plot_spaghetti_with_lmm(
  dt_rt, "log_rt", m_rt, "log RT (s)",
  speed_breaks = speed_breaks, speed_labels = speed_labels
) + labs(tag = "a.")

# Accuracy: separate plot with back-transformed LMM line
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
    n    = sum(!is.na(acc)),
    .groups = "drop"
  ) %>%
  filter(n > 1) %>%   # remove problematic groups entirely
  mutate(
    se = sd / sqrt(n),
    ci = qt(0.975, df = n - 1) * se
  )

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
  theme_minimal(base_size = 18) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor    = element_blank(),
    plot.tag            = element_text(size = 24, face = "bold")
  )

p_fix_spag <- plot_spaghetti_with_lmm(
  dt_eye, "fix_count", m_fix, "Fixation count",
  speed_breaks = speed_breaks, speed_labels = speed_labels
) + labs(tag = "a.")

p_scan_spag <- plot_spaghetti_with_lmm(
  dt_eye, "scanpath", m_scan, "Scanpath length (deg)",
  speed_breaks = speed_breaks, speed_labels = speed_labels
) + labs(tag = "b.")

p_disp_spag <- plot_spaghetti_with_lmm(
  dt_eye, "dispersion", m_disp, "Dispersion (deg²)",
  speed_breaks = speed_breaks, speed_labels = speed_labels
) + labs(tag = "c.")

p_ctr_spag <- plot_spaghetti_with_lmm(
  dt_eye, "center_dist", m_ctr, "Distance from centre (deg)",
  speed_breaks = speed_breaks, speed_labels = speed_labels
) + labs(tag = "d.")

print(p_rt_spag)
print(p_acc_spag)
print(p_fix_spag)
print(p_scan_spag)
print(p_disp_spag)
print(p_ctr_spag)

library(broom.mixed)

models <- list(
  "Reaction time (log s)" = m_rt,
  "Accuracy (logit)"      = m_acc,
  "Fixation count"        = m_fix,
  "Scanpath length (deg)" = m_scan,
  "Dispersion (deg²)"     = m_disp,
  "Centre distance (deg)" = m_ctr
)

results_table <- bind_rows(
  lapply(names(models), function(nm) {
    tidy(models[[nm]], effects = "fixed") %>%
      mutate(outcome = nm) %>%
      select(outcome, term, estimate, std.error, statistic, p.value)
  })
)

# Round for reporting
results_table <- results_table %>%
  mutate(across(c(estimate, std.error, statistic), ~ round(., 3)),
         p.value = ifelse(p.value < .001, "< .001", as.character(round(p.value, 3))))

write.csv(results_table, "results_table.csv", row.names = FALSE)
print(results_table)

