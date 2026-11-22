suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(data.table))
kdm_calc <- BioAge::kdm_calc
hd_calc <- BioAge::hd_calc
out="fits/M7_bioage"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
markers=c("albumin","creatinine","glucose","lncrp","lymph_pct","mcv","rdw","alp","wbc")
prep=function(f) {
  for (n in c(markers,"crp")) if (!n %in% names(f)) f[[n]]=NA_real_
  f$lncrp=ifelse(f$crp>0,log(pmax(f$crp,1e-20)),NA_real_)
  f$sampleID=as.character(f$ID)
  as.data.frame(f)
}
dev=as.data.frame(fread("cache/dev.csv.gz"))
dev$sex=dev$male
dev$ID=dev$SEQN
dev=prep(dev)
fits=list()
for (sex in 0:1) {
  sel=dev$sex==sex & dev$age>=30 & dev$age<=75
  train=dev[which(sel),]
  kdm=kdm_calc(data=train,biomarkers=markers)$fit
  stopifnot(is.finite(kdm$s_ba2), kdm$s_ba2>0)
  # Sex-specific source reference ranges from BioAge/R/hd_nhanes.R.
  ok=dev$sex==sex & dev$age>=20 & dev$age<=30 & dev$bmi<30 &
     (sex==1 | dev$pregnant==2) & between(dev$albumin,3.5,5) &
     between(dev$creatinine,c(.6,.8)[sex+1],c(1.1,1.3)[sex+1]) &
     between(dev$glucose,60,100) & dev$crp<2 &
     between(dev$lymph_pct,20,40) & between(dev$mcv,c(78,82)[sex+1],c(101,102)[sex+1]) &
     between(dev$rdw,11.5,14.5) & between(dev$alp,c(37,45)[sex+1],c(98,115)[sex+1]) &
     between(dev$wbc,4.5,11)
  ref=dev[which(ok & complete.cases(dev[,markers])),,drop=FALSE]
  stopifnot(nrow(ref)>length(markers)+1)
  covm=cov(ref[,markers])
  cen=colMeans(ref[,markers])
  # Original HD standardizes within the projection sample. Fix the divisor in
  # development for comparable sex-specific scores across retrospective cohorts.
  vdev=dev[dev$sex==sex & complete.cases(dev[,markers]),markers]
  hd_scale=sd(sqrt(mahalanobis(vdev,cen,covm)))
  # Cross-check the original calculator on tightly selected reference data.
  direct=sqrt(mahalanobis(ref[,markers],cen,covm))
  projected=hd_calc(ref,ref,markers)$data$hd
  stopifnot(max(abs(projected-direct/sd(direct)))<1e-8)
  fits[[sex+1]]=list(kdm=kdm,mean=cen,cov=covm,scale=hd_scale,hd_n=nrow(ref),
                      kdm_n=nrow(train))
}
saveRDS(fits,file.path(out,"sex_specific_reference.rds"))
dev$hd = NA_real_
for (sex in 0:1) {
  valid = which(dev$sex==sex & complete.cases(dev[,markers]))
  fit=fits[[sex+1]]
  dev$hd[valid]=sqrt(mahalanobis(dev[valid,markers],fit$mean,fit$cov))/fit$scale
}
hd_cox=survival::coxph(survival::Surv(time_yr,event)~age+hd,data=dev,ties="breslow")
fwrite(data.frame(term=names(coef(hd_cox)),coef=as.numeric(coef(hd_cox))),file.path(out,"age_hd_cox.csv"))
fwrite(data.frame(sex=0:1,kdm_n=sapply(fits,"[[","kdm_n"),hd_n=sapply(fits,"[[","hd_n")),
       file.path(out,"reference_counts.csv"))
args=commandArgs(trailingOnly=TRUE)
input_pattern=if(length(args)) paste0("^",args[1],"_.*csv.gz$") else "^(sim|external|actual|primary)_.*csv.gz$"
for (path in list.files("cache",pattern=input_pattern,full.names=TRUE)) {
  d=prep(as.data.frame(fread(path,colClasses=c(ID="character"))))
  result=data.frame(ID=d$ID,M7_KDM=NA_real_,M7_HD=NA_real_,M7_PhenoAge=NA_real_)
  for (sex in 0:1) {
    fit=fits[[sex+1]]
    rows=which(d$sex==sex)
    if (!length(rows)) next
    if(sum(colSums(!is.na(d[rows,markers]))>0)>=length(markers)-2)
        result$M7_KDM[rows]=kdm_calc(data=d[rows,],biomarkers=markers,
                                  fit=fit$kdm,s_ba2=fit$kdm$s_ba2)$data$kdm
    valid=rows[complete.cases(d[rows,markers])]
    if(length(valid))
       result$M7_HD[valid]=sqrt(mahalanobis(d[valid,markers],fit$mean,fit$cov))/fit$scale
  }
  xb=with(d,-19.90667-0.03359355*albumin*10+0.009506491*creatinine*88.42+
        0.1953192*glucose*0.0555+0.09536762*lncrp-0.01199984*lymph_pct+
        0.02676401*mcv+0.3306156*rdw+0.001868778*alp+0.05542406*wbc+0.08035356*age)
  result$M7_PhenoAge=141.50225+(log(.0055305*1.51714/.007692696)+xb)/.090165
  result$M7_age_HD=coef(hd_cox)["age"]*d$age+coef(hd_cox)["hd"]*result$M7_HD
  target=file.path(out,basename(path))
  tmp=file.path(out,paste0("staged_",basename(path)))
  fwrite(result,tmp)
  stopifnot(file.copy(tmp,target,overwrite=TRUE))
  unlink(tmp)
  cat(basename(path), colSums(is.finite(as.matrix(result[,-1]))), "\n")
}
