# Created on Fri Mar  6 16:52:32 2026

# @author: herttaleinonen

# ============================================================
# Visibility functions by speed
# Uses long.csv dataset
# Plots mean accuracy ± SEM for each speed condition
# ============================================================
library(readr)
library(dplyr)
library(ggplot2)
library(patchwork)

# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------
dat <- read_csv("data/long.csv", show_col_types = FALSE)

# ------------------------------------------------------------
# 2) Keep visibility tasks and define speed
# ------------------------------------------------------------
vt <- dat %>%
  filter(task %in% paste0("vt", 1:5)) %>%
  mutate(
    participant = factor(participant),
    speed = case_when(
      task == "vt1" ~ 0.000,
      task == "vt2" ~ 2.703,
      task == "vt3" ~ 5.406,
      task == "vt4" ~ 8.109,
      task == "vt5" ~ 10.812
    ),
    speed_lab = factor(
      c("0", "3", "5.5", "8", "11")[match(task, paste0("vt", 1:5))],
      levels = c("0", "3", "5.5", "8", "11")
    ),
    correct = as.numeric(correct),
    ecc_deg = as.numeric(ecc_deg)
  ) %>%
  filter(!is.na(speed), !is.na(correct), !is.na(ecc_deg))

# ------------------------------------------------------------
# 3) Participant-level accuracy
# ------------------------------------------------------------
vt_participant <- vt %>%
  group_by(participant, speed, speed_lab, ecc_deg) %>%
  summarise(
    acc = mean(correct, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 4) Group mean + SEM
# ------------------------------------------------------------
vt_summary <- vt_participant %>%
  group_by(speed_lab, ecc_deg) %>%
  summarise(
    mean_acc = mean(acc, na.rm = TRUE),
    sem      = sd(acc, na.rm = TRUE) / sqrt(n()),
    .groups  = "drop"
  )

# ------------------------------------------------------------
# 5) Journal-compliant theme 
# ------------------------------------------------------------
journal_theme <- theme_minimal(base_size = 18, base_family = "Helvetica") +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor    = element_blank(),
    axis.text           = element_text(size = 14, face = "plain", family = "Helvetica"),
    axis.title          = element_text(size = 18, face = "plain", family = "Helvetica"),
    legend.text         = element_text(size = 14, face = "plain", family = "Helvetica"),
    legend.title        = element_text(size = 14, face = "plain", family = "Helvetica"),
    plot.tag            = element_text(size = 24, face = "plain", family = "Helvetica"),
    axis.ticks          = element_line(color = "#333333"),
    axis.ticks.length   = unit(3, "pt"),
    axis.line           = element_line(color = "#333333", linewidth = 0.5),
    plot.margin         = margin(6, 8, 6, 6)
  )

# ------------------------------------------------------------
# 6) Plot
# ------------------------------------------------------------
p_visibility <- ggplot(
  vt_summary,
  aes(x = ecc_deg, y = mean_acc, color = speed_lab, group = speed_lab)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  geom_errorbar(
    aes(ymin = mean_acc - sem, ymax = mean_acc + sem),
    width = 0.2
  ) +
  scale_y_continuous(
    breaks = seq(0.6, 1, 0.1),
    limits = c(0.6, 1)
  ) +
  labs(
    x     = "Eccentricity (deg)",
    y     = "Accuracy",
    color = "Speed (deg/s)",
    tag   = "a."
  ) +
  journal_theme

print(p_visibility)

# ------------------------------------------------------------
# 8) Plot: accuracy vs speed at 16° eccentricity
# Shows the inverted-U relationship
# ------------------------------------------------------------
vt_16 <- vt_participant %>%
  filter(ecc_deg == 16)

vt_16_summary <- vt_16 %>%
  group_by(speed, speed_lab) %>%
  summarise(
    mean_acc = mean(acc, na.rm = TRUE),
    sem      = sd(acc, na.rm = TRUE) / sqrt(n()),
    .groups  = "drop"
  )

p_16deg <- ggplot(vt_16_summary,
                  aes(x = speed, y = mean_acc)) +
  geom_line(linewidth = 1.2, color = "#333333") +
  geom_point(size = 2.5, color = "#333333") +
  geom_errorbar(
    aes(ymin = mean_acc - sem, ymax = mean_acc + sem),
    width = 0.25, color = "#333333"
  ) +
  scale_x_continuous(
    breaks = c(0.000, 2.703, 5.406, 8.109, 10.812),
    labels = c("0", "3", "5.5", "8", "11")
  ) +
  scale_y_continuous(
    breaks = seq(0.5, 1, 0.1),
    limits = c(0.6, 1)
  ) +
  labs(
    x   = "Velocity (deg/s)",
    y   = "Accuracy",
    tag = "b."
  ) +
  journal_theme

print(p_16deg)

# ------------------------------------------------------------
# 9) Save figures
# ------------------------------------------------------------
ggsave("visibility_functions_by_speed.pdf", p_visibility,
       width = 7, height = 5, device = cairo_pdf)
ggsave("visibility_functions_by_speed.eps", p_visibility,
       width = 7, height = 5, device = "eps")
ggsave("visibility_functions_by_speed.png", p_visibility,
       width = 7, height = 5, dpi = 300)

ggsave("visibility_16deg.pdf", p_16deg,
       width = 7, height = 5, device = cairo_pdf)
ggsave("visibility_16deg.eps", p_16deg,
       width = 7, height = 5, device = "eps")
ggsave("visibility_16deg.png", p_16deg,
       width = 7, height = 5, dpi = 300)

# Combined side-by-side figure
fig_vis <- p_visibility + p_16deg +
  plot_layout(ncol = 2)
ggsave("figure_visibility_combined.pdf", fig_vis,
       width = 14, height = 5, device = cairo_pdf)
ggsave("figure_visibility_combined.eps", fig_vis,
       width = 14, height = 5, device = "eps")
ggsave("figure_visibility_combined.png", fig_vis,
       width = 14, height = 5, dpi = 300)

cat("All visibility figures saved.\n")
