suppressPackageStartupMessages({
  library(data.table)
  library(survival)
})
setDTthreads(2)
args <- commandArgs(trailingOnly = TRUE)
coverage <- c(0, 7, 30, 90, 180)
outcomes <- c("hearte", "stroke")
panels <- c("liver", "kidney", "metabolic", "immune_heme", "cardiovascular")
base <- "age10 + I(age10^2) + sex + factor(education) + full_z"
fit <- function(data, rhs) {
  coxph(as.formula(paste("Surv(t, status == 1) ~", rhs)),
        data = data, ties = "efron", x = TRUE)
}
household <- function(d, cohort) {
  if (cohort == "hrs") return(sub("-[0-9]{3}$", "", d$ID))
  d$ID
}

hrs <- fread(file.path(args[1], "hrs_events.csv"), colClasses = c(ID = "character"))
subtypes <- c("hrtatte", "conhrtfe", "angine", "hrtrhme")
cells <- rbindlist(list(CJ(cohort = "hrs", endpoint = outcomes, days = coverage, family = "primary"),
                       CJ(cohort = "hrs", endpoint = subtypes, days = c(30, 90, 180), family = "subtypes"),
                       CJ(cohort = "elsa", endpoint = subtypes, days = 0, family = "subtypes")))
result <- list()
for (i in seq_len(nrow(cells))) {
  cell <- cells[i]
  d <- hrs[cohort == cell$cohort & endpoint == cell$endpoint & lookback_days == cell$days]
  available <- if (cell$cohort == "hrs") panels else c("metabolic", "immune_heme", "cardiovascular")
  n <- nrow(d)
  evt <- sum(d$status == 1)
  row <- list(cohort = cell$cohort, family = cell$family, endpoint = cell$endpoint, lookback_days = cell$days,
              n = n, events = evt,
              competing_deaths = sum(d$status == 2), status = "less_than_20_events")
  if (evt >= 20) {
    m <- lapply(available, function(group) fit(d, paste(base, "+", group)))
    beta <- setNames(sapply(seq_along(m), function(i) coef(m[[i]])[[available[i]]]), available)
    changes <- sapply(m, function(model) residuals(model, type = "dfbeta")[, ncol(model$x)])
    clusters <- rowsum(changes, household(d, cell$cohort))
    j <- match("cardiovascular", available)
    delta <- beta[j] - beta[-j]
    cluster_deltas <- clusters[, j] - clusters[, -j, drop = FALSE]
    se <- sqrt(colSums(cluster_deltas^2))
    matched_se <- sqrt(sum(clusters[, j]^2))
    row$status <- "estimated"
    row$hr_cardio <- exp(beta[j])
    row$hr_low <- exp(beta[j] - 1.96 * matched_se)
    row$hr_high <- exp(beta[j] + 1.96 * matched_se)
    row$strongest_other <- names(which.max(beta[-j]))
    row$hr_strongest_other <- exp(max(beta[-j]))
    row$p_iut <- max(pnorm(delta / se, lower.tail = FALSE))
    row$minimum_logHR_advantage <- min(delta)
  }
  result[[length(result) + 1]] <- row
}
cardio <- rbindlist(result, fill = TRUE)
cardio[, q_iut := p.adjust(p_iut, method = "BH"), by = family]
fwrite(cardio[family == "primary"], file.path(args[2], "hrs_incident_cardiovascular.csv"))
fwrite(cardio[family == "subtypes"], file.path(args[2], "cardiovascular_disease_subtypes.csv"))

rows <- fread(file.path(args[1], "markers.csv"), colClasses = c(ID = "character"))
result <- list()
for (i in seq_len(nrow(unique(rows[, .(cohort, endpoint)])))) {
  cell <- unique(rows[, .(cohort, endpoint)])[i]
  d <- rows[cohort == cell$cohort & endpoint == cell$endpoint]
  stopifnot(!anyDuplicated(d$ID))
  d[, family := household(d, cell$cohort)]
  d[, marker_z := as.numeric(scale(marker))]
  n <- nrow(d)
  evt <- sum(d$status == 1)
  stopifnot(evt >= 20)
  set.seed(1471822)
  families <- unique(d$family)
  fold_map <- setNames(sample(rep(seq_len(5), length.out = length(families))), families)
  d[, fold := fold_map[family]]
  stopifnot(all(d[status == 1, .N, by = fold]$N >= 2), uniqueN(d$fold) == 5)
  predictors <- c(baseline = "", marker = "+ marker_z", organ = "+ group_z",
                  marker_organ = "+ marker_z + group_z")
  by_fold <- lapply(seq_len(5), function(k) {
    train <- d[fold != k]
    held <- d[fold == k]
    for (key in names(predictors)) {
      m <- fit(train, paste(base, predictors[[key]]))
      held[[key]] <- predict(m, held, type = "lp")
    }
    cstat <- concordance(Surv(t, status == 1) ~ baseline + marker + organ + marker_organ,
                        data = held, cluster = family, reverse = TRUE)
    comparable <- rowSums(cstat$count[, c("concordant", "discordant", "tied.x"), drop = FALSE])
    n_comp <- unname(comparable[1])
    stopifnot(all(comparable == n_comp))
    list(C = cstat$concordance, variance = cstat$var, n_comp = n_comp)
  })
  weights <- sapply(by_fold, function(x) x$n_comp)
  weights <- weights / sum(weights)
  estimate <- Reduce("+", Map(function(x, w) x$C * w, by_fold, weights))
  covar <- Reduce("+", Map(function(x, w) x$variance * w^2, by_fold, weights))
  contrast <- c(0, -1, 0, 1)
  diff <- sum(estimate * contrast)
  se <- sqrt(drop(t(contrast) %*% covar %*% contrast))
  result[[length(result) + 1]] <- list(cohort = cell$cohort, endpoint = cell$endpoint,
    n = n, events = evt, competing_deaths = sum(d$status == 2), organ = d$group[1],
    comparator = d$marker_name[1], baseline_C = estimate[1], plus_marker_C = estimate[2],
    plus_organ_C = estimate[3], plus_both_C = estimate[4], delta_C_beyond_marker = diff,
    lo = diff - 1.96 * se, hi = diff + 1.96 * se, p = 2 * pnorm(-abs(diff / se)))
}
comparisons <- rbindlist(result)
comparisons[, q := p.adjust(p, method = "BH")]
fwrite(comparisons, file.path(args[2], "community_marker_comparisons.csv"))
