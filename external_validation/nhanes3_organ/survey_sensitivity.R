# Design-based cause-specific hazard domain analysis with the MEC exam weights.
args <- commandArgs(trailingOnly = TRUE)
extra_library <- Sys.getenv("NH3_SURVEY_LIBRARY")
if (nzchar(extra_library)) .libPaths(c(extra_library, .libPaths()))
suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(survival))

all_mec <- read.csv(args[[1]])
all_mec$in_domain <- tolower(as.character(all_mec$in_domain)) == "true"
design <- svydesign(
  ids = ~ SDPPSU6, strata = ~ SDPSTRA6, weights = ~ WTPFEX6,
  data = all_mec, nest = TRUE
)
domain <- subset(design, in_domain)
design_df <- degf(domain)
panels <- c("cardiovascular", "kidney", "metabolic", "immune_heme")
rows <- lapply(panels, function(panel) {
  covariate <- paste0(panel, "_sd")
  event <- paste0(panel, "_death")
  formula <- reformulate(
    c("age_linear", "age_quadratic", "sex", "full_sd", covariate),
    sprintf("Surv(time_yr, %s)", event)
  )
  model <- svycoxph(formula, design = domain)
  stopifnot(identical(model$method, "efron"))
  beta <- unname(coef(model)[covariate])
  se <- sqrt(vcov(model)[covariate, covariate])
  half_width <- qt(.975, df = design_df) * se
  data.frame(
    analysis = "NHANES_III_design_weighted", panel = panel,
    n = nrow(domain$variables), events = sum(domain$variables[[event]]),
    HR_per_SD = exp(beta), CI95_lower = exp(beta - half_width),
    CI95_upper = exp(beta + half_width),
    p = 2 * pt(-abs(beta / se), df = design_df),
    standard_error = se, design_df = design_df
  )
})
write.csv(do.call(rbind, rows), args[[2]], row.names = FALSE)
