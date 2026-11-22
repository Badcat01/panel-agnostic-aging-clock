suppressPackageStartupMessages({
  library(data.table)
  library(survival)
})
args <- commandArgs(trailingOnly = TRUE)
prior <- normalizePath(args[1])
output <- normalizePath(args[2])
tmp <- normalizePath(args[3])
setDTthreads(2)
load_table <- function(path) {
  conn <- if (endsWith(path, ".gz")) gzfile(path, open = "rt") else file(path, open = "rt")
  on.exit(close(conn))
  as.data.table(read.csv(conn, colClasses = c(ID = "character")))
}
method <- "age10 + I(age10^2) + sex + education"
bootstrap_count <- 2000

event_set <- function(base, visits, disease, eligible, landmark = "index") {
  if (landmark == "index") base[, time_yr := 0]
  follow <- merge(visits[!is.na(get(disease)) & is.finite(time)], base, by = "ID")
  follow <- follow[time > time_yr & time <= horizon & (death == 0 | time <= death_upper + 0.003)]
  history <- if (!nrow(follow)) data.table(ID = character(), pos = numeric(), last = numeric()) else follow[, .(
    pos = if (any(get(disease) == 1)) min(time[get(disease) == 1]) else Inf,
    last = max(time)
  ), by = ID]
  d <- merge(base[ID %chin% eligible], history, by = "ID", all.x = TRUE)
  d[, status := fcase(is.finite(pos), 1L, death == 1 & death_or_censor <= horizon, 2L, default = 0L)]
  d[, t := fcase(status == 1, pos, status == 2, death_or_censor, default = last) - time_yr]
  d[is.finite(t) & t > 0]
}

cox <- function(d, covariates) {
  coxph(as.formula(paste("Surv(t, status == 1) ~", covariates)), data = d,
        ties = "efron", model = TRUE)
}
coef_row <- function(fit, term, extra) {
  est <- unname(coef(fit)[term])
  se <- sqrt(vcov(fit)[term, term])
  ph <- cox.zph(fit)$table[term, "p"]
  c(extra, list(HR = exp(est), lo = exp(est - 1.96 * se), hi = exp(est + 1.96 * se),
                p = 2 * pnorm(-abs(est / se)), ph_p = ph))
}

cv_compare <- function(d, base_rhs, term, seed) {
  # HRS households stay within a fold. For other cohorts use individuals.
  folds <- as.data.table(d)[, .(events = sum(status == 1)), by = group_id]
  set.seed(seed)
  folds[, fold := sample(rep(seq_len(5), length.out = .N)), by = (events > 0)]
  d <- merge(d, folds[, .(group_id, fold)], by = "group_id")
  scores <- vector("list", 5)
  for (k in seq_len(5)) {
    train <- d[fold != k]
    test <- d[fold == k]
    fit0 <- cox(train, base_rhs)
    fit1 <- cox(train, paste(base_rhs, "+", term))
    stopifnot(all(is.finite(coef(fit0))), all(is.finite(coef(fit1))))
    test[, `:=`(base_lp = predict(fit0, newdata = test, type = "lp"),
                augmented_lp = predict(fit1, newdata = test, type = "lp"),
                event = status == 1, cluster_index = match(group_id, unique(group_id)))]
    scores[[k]] <- test
  }
  count <- function(d, lp, freq) {
    risk <- d[[lp]]
    eligible <- freq > 0
    cc <- concordancefit(Surv(d$t[eligible], d$event[eligible]), risk[eligible],
                         weights = freq[eligible], reverse = TRUE, timewt = "n", std.err = FALSE)$count
    c(cc[1] + 0.5 * cc[3], sum(cc[1:3]))
  }
  calc <- function(random) {
    pooled <- numeric(4)
    for (d in scores) {
      size <- max(d$cluster_index)
      freq <- if (random) tabulate(sample.int(size, size, replace = TRUE), size)[d$cluster_index] else rep(1, nrow(d))
      pooled <- pooled + c(count(d, "base_lp", freq), count(d, "augmented_lp", freq))
    }
    c(pooled[1] / pooled[2], pooled[3] / pooled[4])
  }
  observed <- calc(FALSE)
  deltas <- replicate(bootstrap_count, {
    pair <- calc(TRUE)
    pair[2] - pair[1]
  })
  ci <- quantile(deltas, c(0.025, 0.975))
  list(base_C = observed[1], augmented_C = observed[2], delta_C = observed[2] - observed[1],
       delta_lo = unname(ci[1]), delta_hi = unname(ci[2]), bootstrap = bootstrap_count)
}

health <- load_table(file.path(tmp, "hrs_cognition.csv"))
hrs <- load_table(file.path(prior, "hrs_baseline.csv.gz"))
hrs[, `:=`(age10 = (age - 60) / 10, education = factor(education),
           group_id = sub("-[0-9]+$", "", ID))]
health_rows <- list()
for (scheme in c("self_only", "observed_proxy_plus_self")) {
  wave <- copy(health)
  wave[, class := if (scheme == "self_only") self_class else combined_class]
  for (endpoint in c("screen_dementia", "cognitive_impairment_or_dementia")) {
    threshold <- if (endpoint == "screen_dementia") 2 else 1
    wave[, binary := as.integer(class >= threshold)]
    eligible <- wave[wave == 13 & binary == 0, ID]
    d <- event_set(copy(hrs), wave[wave > 13], "binary", eligible)
    d <- d[complete.cases(age10, sex, education, fi0, full_z, ridge_lp)]
    stopifnot(sum(d$status == 1) > 20, !anyDuplicated(d$ID))
    cc <- cv_compare(d, paste(method, "+ fi0"), "full_z", 20261125)
    for (adjusted in c(FALSE, TRUE)) {
      rhs <- paste(method, "+ full_z", if (adjusted) "+ fi0" else "", "+ cluster(group_id)")
      estimate <- coef_row(cox(d, rhs), "full_z",
                           list(scheme = scheme, endpoint = endpoint, adjusted_for_FI = adjusted,
                                n = nrow(d), events = sum(d$status == 1),
                                competing_deaths = sum(d$status == 2)))
      if (adjusted) estimate <- c(estimate, cc)
      health_rows[[length(health_rows) + 1]] <- estimate
    }
  }
}
health_results <- rbindlist(health_rows, fill = TRUE)
health_results[, q := p.adjust(p, "BH"), by = scheme]
fwrite(health_results, file.path(output, "hrs_proxy_healthspan.csv"))

physical <- load_table(file.path(tmp, "hrs_physical.csv"))
physical <- merge(physical, hrs, by = "ID")
physical <- physical[is.finite(follow_2018_years) & follow_2018_years > 0 &
                       interval_2014_2018 > follow_2018_years & follow_2018_years <= horizon &
                       (death == 0 | follow_2018_years <= death_upper + 0.003)]
phys_rows <- list()
for (measure in c("grp", "timwlk")) {
  d <- copy(physical)
  d[, `:=`(baseline_measure = get(paste0(measure, "_12")), y = get(paste0(measure, "_14")))]
  d <- d[complete.cases(age10, sex, education, fi0, full_z, baseline_measure, y, follow_2018_years, interval_2014_2018)]
  if (nrow(d) < 100) next
  if (measure == "timwlk") d[, `:=`(baseline_measure = log(baseline_measure), y = log(y))]
  for (adjusted in c(FALSE, TRUE)) {
    rhs <- paste(method, "+ baseline_measure + follow_2018_years + interval_2014_2018 + full_z",
                 if (adjusted) "+ fi0" else "")
    fit <- lm(as.formula(paste("y ~", rhs)), data = d, x = TRUE)
    x <- fit$x
    bread <- solve(crossprod(x))
    cluster_score <- rowsum(x * resid(fit), d$group_id)
    clusters <- nrow(cluster_score)
    covariance <- bread %*% crossprod(cluster_score) %*% bread * (clusters / (clusters - 1)) *
      ((nrow(x) - 1) / (nrow(x) - ncol(x)))
    se <- sqrt(covariance["full_z", "full_z"])
    b <- unname(coef(fit)["full_z"])
    ci <- b + c(-1, 1) * qt(.975, clusters - 1) * se
    phys_rows[[length(phys_rows) + 1]] <- list(measure = measure, adjusted_for_FI = adjusted,
      n = nrow(d), n_households = clusters, beta = b, lo = ci[1], hi = ci[2],
      p = 2 * pt(-abs(b / se), clusters - 1), r_squared = summary(fit)$r.squared,
      q99_abs_studentized_resid = unname(quantile(abs(rstandard(fit)), 0.99)))
  }
}
phys_results <- rbindlist(phys_rows)
if (nrow(phys_results)) phys_results[, q := p.adjust(p, "BH")]
fwrite(phys_results, file.path(output, "hrs_physical_2014_2018.csv"))

speed_rows <- list()
for (cohort in c("charls", "elsa")) {
  prefix <- if (cohort == "charls") "charls_w5" else "elsa"
  b <- load_table(file.path(prior, paste0(prefix, "_baseline.csv.gz")))
  l <- load_table(file.path(prior, paste0(prefix, "_visits.csv.gz")))
  for (w in if (cohort == "charls") 3 else c(4, 6)) {
    pace <- load_table(file.path(prior, paste0(cohort, "_", w, "_pace.csv.gz")))
    pp <- merge(pace, b, by = "ID")
    pp[, `:=`(age10 = (landmark_age - 60) / 10, education = factor(education),
             recent_ba = baseline_ba + pace * time_yr, group_id = ID)]
    stem <- paste(method, "+ time_yr")
    for (ep in c("death", "adl")) {
      if (ep == "death") {
        d <- copy(pp)[, `:=`(t = death_or_censor - time_yr, status = death)]
      } else {
        eligible <- l[wave <= w, .(eligible = any(wave == w & adl == 0, na.rm = TRUE) &
                                  !any(adl == 1, na.rm = TRUE)), by = ID][eligible == TRUE, ID]
        d <- event_set(copy(pp), l[wave > w], "adl", eligible, "second")
      }
      d <- d[is.finite(t) & t > 0 & complete.cases(age10, sex, education, pace, recent_ba)]
      info <- list(cohort = cohort, second_wave = w, endpoint = ep, n = nrow(d),
                   events = sum(d$status == 1), competing_deaths = sum(d$status == 2))
      if (info$events < 20) {
        speed_rows[[length(speed_rows) + 1]] <- c(info, list(status = "too_few_events"))
        next
      }
      rhs <- paste(stem, "+ recent_ba")
      cc <- cv_compare(d, rhs, "pace", 20261125)
      estimate <- coef_row(cox(d, paste(rhs, "+ pace")), "pace", c(info, list(status = "estimated")))
      speed_rows[[length(speed_rows) + 1]] <- c(estimate, list(score_pace_pearson = cor(d$pace, d$recent_ba)), cc)
    }
  }
}
speed_results <- rbindlist(speed_rows, fill = TRUE)
speed_results[, q := p.adjust(p, "BH")]
fwrite(speed_results, file.path(output, "pace_beyond_latest_score.csv"))

# W7-W9 interviews extend W6 functional follow-up. Their death data are
# incomplete, so condition this analysis on returning for at least one wave.
l <- load_table(file.path(prior, "elsa_visits_w2_to_w9.csv.gz"))
pace <- load_table(file.path(prior, "elsa_6_pace.csv.gz"))
pp <- merge(pace, b, by = "ID")
pp[, `:=`(age10 = (landmark_age - 60) / 10, education = factor(education),
          recent_ba = baseline_ba + pace * time_yr, group_id = ID)]
dated <- merge(l[, .(ID, wave, time, adl, iadl)], pp, by = "ID")
rows <- list()
for (ep in c("adl", "iadl")) {
  eligible <- dated[wave <= 6 & time <= time_yr, .(
    eligible = any(wave == 6 & get(ep) == 0, na.rm = TRUE) &
               !any(get(ep) == 1, na.rm = TRUE)
  ), by = ID][eligible == TRUE, ID]
  follow <- dated[wave > 6 & time > time_yr & ID %chin% eligible &
                    !is.na(get(ep)) & (death == 0 | time <= death_upper + .003)]
  history <- follow[, .(
    t = if (any(get(ep) == 1)) min(time[get(ep) == 1]) - time_yr[1] else max(time) - time_yr[1],
    status = as.integer(any(get(ep) == 1))
  ), by = ID]
  d <- merge(pp, history, by = "ID")[complete.cases(age10, sex, education, pace, recent_ba)]
  rhs <- paste(method, "+ time_yr + recent_ba")
  cc <- cv_compare(d, rhs, "pace", 20261125)
  estimate <- coef_row(cox(d, paste(rhs, "+ pace")), "pace",
                       list(cohort = "elsa", second_wave = 6, endpoint = ep,
                            first_wave = 2, last_outcome_wave = 9,
                            n = nrow(d), events = sum(d$status == 1), returned_respondents_only = TRUE))
  rows[[length(rows) + 1]] <- c(estimate, cc)
}
extended_results <- rbindlist(rows, fill = TRUE)
extended_results[, q := p.adjust(p, "BH")]
fwrite(extended_results, file.path(output, "elsa_w6_observed_speed_increment.csv"))
