#!/usr/bin/env Rscript

# Eoin-style validation for WLNM_dir_neg Apocrita.
# Requires the CSVs produced by prepare_eoin_wlnm_dir_neg_apocrita_train90.py.

args <- commandArgs(trailingOnly = TRUE)

default_output_dir <- file.path(
  "src", "matlab", "data",
  "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita",
  "statistical_tests", "eoin"
)

output_dir <- if (length(args) >= 1) args[[1]] else default_output_dir
train_ratio <- if (length(args) >= 2) args[[2]] else "90"
train_ratio_number <- suppressWarnings(as.numeric(train_ratio))
if (!is.na(train_ratio_number) && train_ratio_number <= 1) {
  train_ratio_number <- train_ratio_number * 100
}
if (!is.na(train_ratio_number) && abs(train_ratio_number - round(train_ratio_number)) <= 1e-9) {
  train_suffix <- paste0("train", as.integer(round(train_ratio_number)))
} else {
  train_suffix <- paste0("train", gsub("\\.", "p", train_ratio))
}
input_file <- file.path(output_dir, paste0("eoin_mixed_model_long_", train_suffix, ".csv"))

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

if (!requireNamespace("nlme", quietly = TRUE)) {
  stop("Package 'nlme' is required. Install it with install.packages('nlme').")
}

has_emmeans <- requireNamespace("emmeans", quietly = TRUE)

data <- read.csv(input_file, stringsAsFactors = FALSE)
if ("Foodweb" %in% names(data)) data$web <- data$Foodweb
if ("WebType" %in% names(data)) data$web_type <- data$WebType
if ("EcosystemType" %in% names(data)) data$ecosystem <- data$EcosystemType
if ("Metric" %in% names(data)) data$metric <- data$Metric
if ("Value" %in% names(data)) data$value <- data$Value

required_columns <- c("web", "web_type", "ecosystem", "metric", "value")
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop("Input table is missing columns: ", paste(missing_columns, collapse = ", "))
}

data$web <- factor(data$web)
data$web_type <- factor(data$web_type, levels = c("real", "pseudo"))
data$ecosystem <- factor(data$ecosystem)
data$metric <- factor(data$metric)

metrics <- unique(as.character(data$metric))

paired_rows <- list()
anova_rows <- list()
coefficient_rows <- list()
posthoc_rows <- list()
lme_control <- nlme::lmeControl(opt = "optim", msMaxIter = 200, msMaxEval = 200)

for (metric_name in metrics) {
  metric_data <- data[data$metric == metric_name, ]
  id_columns <- c("web", "ecosystem")
  if ("run_id" %in% names(metric_data)) {
    id_columns <- c("web", "run_id", "ecosystem")
  }
  wide_input <- metric_data[, c(id_columns, "web_type", "value")]
  wide_input$web_type <- as.character(wide_input$web_type)
  wide <- reshape(
    wide_input,
    idvar = id_columns,
    timevar = "web_type",
    direction = "wide"
  )

  t_result <- t.test(wide$value.pseudo, wide$value.real, paired = TRUE)
  paired_rows[[as.character(metric_name)]] <- data.frame(
    metric = as.character(metric_name),
    n_pairs = nrow(wide),
    n_webs = length(unique(wide$web)),
    mean_real = mean(wide$value.real, na.rm = TRUE),
    mean_pseudo = mean(wide$value.pseudo, na.rm = TRUE),
    delta_pseudo_minus_real = mean(wide$value.pseudo - wide$value.real, na.rm = TRUE),
    t_statistic = unname(t_result$statistic),
    df = unname(t_result$parameter),
    p_value = t_result$p.value,
    conf_low = unname(t_result$conf.int[1]),
    conf_high = unname(t_result$conf.int[2]),
    stringsAsFactors = FALSE
  )

  model <- nlme::lme(
    value ~ web_type * ecosystem,
    random = ~ 1 | web,
    data = metric_data,
    method = "REML",
    na.action = na.omit,
    control = lme_control
  )

  anova_table <- as.data.frame(anova(model))
  anova_table$Effect <- row.names(anova_table)
  row.names(anova_table) <- NULL
  anova_table$Metric <- as.character(metric_name)
  anova_rows[[as.character(metric_name)]] <- anova_table[
    , c("Metric", "Effect", setdiff(names(anova_table), c("Metric", "Effect")))
  ]

  coef_table <- as.data.frame(summary(model)$tTable)
  coef_table$Coefficient <- row.names(coef_table)
  row.names(coef_table) <- NULL
  coef_table$Metric <- as.character(metric_name)
  coefficient_rows[[as.character(metric_name)]] <- coef_table[
    , c("Metric", "Coefficient", setdiff(names(coef_table), c("Metric", "Coefficient")))
  ]

  if (has_emmeans) {
    emm <- emmeans::emmeans(model, ~ web_type | ecosystem)
    contrast_table <- as.data.frame(
      emmeans::contrast(emm, method = "pairwise", adjust = "tukey")
    )
    contrast_table$Metric <- as.character(metric_name)
    contrast_table$DeltaPseudoMinusReal <- -contrast_table$estimate
    contrast_table$Direction <- ifelse(
      contrast_table$DeltaPseudoMinusReal > 0,
      "pseudo_higher",
      ifelse(contrast_table$DeltaPseudoMinusReal < 0, "pseudo_lower", "no_mean_difference")
    )
    posthoc_rows[[as.character(metric_name)]] <- contrast_table[
      , c(
        "Metric",
        "ecosystem",
        "contrast",
        "estimate",
        "DeltaPseudoMinusReal",
        "SE",
        "df",
        "t.ratio",
        "p.value",
        "Direction"
      )
    ]
  }
}

paired_output <- do.call(rbind, paired_rows)
anova_output <- do.call(rbind, anova_rows)
coefficient_output <- do.call(rbind, coefficient_rows)
posthoc_output <- if (length(posthoc_rows) > 0) do.call(rbind, posthoc_rows) else NULL

write.csv(
  paired_output,
  file.path(output_dir, paste0("eoin_r_paired_ttest_general_results_", train_suffix, ".csv")),
  row.names = FALSE
)
write.csv(
  anova_output,
  file.path(output_dir, paste0("eoin_r_lme_anova_", train_suffix, ".csv")),
  row.names = FALSE
)
write.csv(
  coefficient_output,
  file.path(output_dir, paste0("eoin_r_lme_coefficients_", train_suffix, ".csv")),
  row.names = FALSE
)
if (!is.null(posthoc_output)) {
  write.csv(
    posthoc_output,
    file.path(output_dir, paste0("eoin_r_lme_posthoc_webtype_by_ecosystem_", train_suffix, ".csv")),
    row.names = FALSE
  )
} else {
  writeLines(
    "Package 'emmeans' was not available, so post-hoc WebType contrasts were not generated.",
    file.path(output_dir, paste0("eoin_r_lme_posthoc_not_generated_", train_suffix, ".txt"))
  )
}

sink(file.path(output_dir, paste0("eoin_r_session_info_", train_suffix, ".txt")))
print(sessionInfo())
cat("\nPackage availability:\n")
cat("emmeans:", has_emmeans, "\n")
sink()

cat("Wrote R validation outputs to ", output_dir, "\n", sep = "")
