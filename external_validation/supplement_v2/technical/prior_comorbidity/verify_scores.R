# Independently compare direct patient-level Quan mapping with the code-dictionary shortcut.
Sys.setlocale("LC_CTYPE",".UTF-8"); .libPaths(c("R_lib",.libPaths()))
suppressPackageStartupMessages({library(data.table);library(comorbidity);library(jsonlite)})
setDTthreads(2)
codes=unique(fread("cache/prior_codes.csv.gz",select=c("subject_id","icd_code","icd_version"),
                  colClasses=c(icd_code="character")))
ids=codes[,.(versions=paste(sort(unique(icd_version)),collapse="+")),by=subject_id]
set.seed(73125)
ids=ids[,.(subject_id=sample(subject_id,min(.N,120))),by=versions]
x=codes[subject_id%in%ids$subject_id]
production=fread("cache/prior_scores.csv.gz")[subject_id%in%ids$subject_id]
for(kind in c("charlson","elixhauser")) {
  dom=rbindlist(lapply(c(9,10),function(i){
    subset=x[icd_version==i]
    as.data.table(comorbidity(subset,id="subject_id",code="icd_code",map=paste0(kind,"_icd",i,"_quan"),assign0=FALSE,labelled=FALSE))
  }))
  merged=as.data.frame(dom[,lapply(.SD,max),by=subject_id])
  class(merged)=c("comorbidity","data.frame");attr(merged,"map")=paste0(kind,"_icd10_quan")
  direct=score(merged,weights=if(kind=="charlson")"charlson" else "vw",assign0=TRUE)
  stopifnot(all(direct==production[[kind]][match(merged$subject_id,production$subject_id)]))
}
write_json(list(sample_by_version=ids[,.N,by=versions],ppl=nrow(ids),both_indices_identical=TRUE,
           configured_package=as.character(packageVersion("comorbidity"))),
           "patient_level_score_check.json",auto_unbox=TRUE,pretty=TRUE)
