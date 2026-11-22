# Usage: Rscript run.R <prepared.csv.gz> <output directory> <synthetic|hrs> [replicates]
# Takes one QC-annotated row per person, including both clocks for a matched comparison.
Sys.setlocale("LC_CTYPE",".UTF-8")
suppressPackageStartupMessages({library(survival);library(data.table);library(jsonlite)})
setDTthreads(1)
arg=commandArgs(trailingOnly=TRUE)
stopifnot(length(arg)%in%c(3,4),arg[3]%in%c("synthetic","hrs"))
real=arg[3]=="hrs"
B=if(length(arg)==4) as.integer(arg[4]) else 2000L
stopifnot(B>=2L, !real || B==2000L)
out=arg[2]
dir.create(out,showWarnings=FALSE,recursive=TRUE)
d=fread(arg[1],colClasses=c(ID="character"))
need=c("ID","age","sex","time_yr","event","clinical","qc_pass")
stopifnot(all(need%in%names(d)),!anyDuplicated(d$ID),
          all(grepl("^[0-9]{6}-[0-9]{3}$",d$ID)), all(na.omit(d$qc_pass)%in%0:1),
          all(na.omit(d$event)%in%0:1),all(na.omit(d$sex)%in%0:1))
if(real) {
  manifest=fromJSON(paste0(arg[1],".json"))
  stopifnot(manifest$type=="HRS-authorized-paired",manifest$prepared_sha256==digest::digest(file=arg[1],algo="sha256"),
            manifest$model_freeze_sha256=="a472b93f36c8d49de51d2c55c720aa801fa0e08d65c71575986e3bd995b20ae1")
}
clocks=grep("^clock_",names(d),value=TRUE)
stopifnot(length(clocks)>0)
d[,family:=sub("-.*","",ID)]
source_rows=nrow(d)
# Draw survey replicates from the sampling frame before taking the analytic domain.
survey_frame=NULL
if(all(c("weight","stratum","psu")%in%names(d))) {
  survey_frame=unique(d[is.finite(weight) & weight>0 & !is.na(stratum) & !is.na(psu),
                        .(unit=paste(stratum,psu,sep="/"),stratum)])
}
d=d[qc_pass==1 & !is.na(qc_pass) & age>=50 & time_yr>0 & is.finite(time_yr) & is.finite(clinical) & !is.na(sex) & !is.na(event)]
d[,age10:=(age-65)/10]
# Balance household folds once, stratifying the households on whether any member died.
fams=d[,.(anydeath=as.integer(any(event==1))),by=family][order(family)]
set.seed(734101)
fams[,fold:=sample(rep(seq_len(5),length.out=.N)),by=anydeath]
d=fams[d,on="family"]
stopifnot(d[,uniqueN(fold),by=family][,all(V1==1)])
point=list(); effects=list(); tests=list(); avail=list(); second=list(); corr=list(); organreg=list()
add_avail=function(label,n,reason) avail[[length(avail)+1]] <<- list(analysis=label,n=n,reason=reason)
ctrl=coxph.control(iter.max=35)

model_lp=function(z,tr,va,terms,w) {
  X=as.matrix(z[,..terms])
  f=coxph.fit(X[tr,,drop=FALSE],Surv(z$time_yr[tr],z$event[tr]),NULL,rep(0,length(tr)),
              NULL,ctrl,w[tr],"breslow",seq_along(tr),resid=FALSE)
  if(any(!is.finite(f$coefficients))) stop("Nonestimable Cox covariates")
  as.numeric(X[va,,drop=FALSE]%*%f$coefficients)
}
pair_harrell=function(t,e,lp,w) {
  res=concordancefit(Surv(t,e),lp,NULL,w,reverse=TRUE,std.err=FALSE)
  tot=sum(res$count[1:3])
  if(tot==0) return(c(win=0,total=0))
  c(win=as.numeric(res$concordance)*tot,total=tot)
}
pair_uno=function(z,tr,va,lp,w) {
  g=survfit(Surv(time_yr,1-event)~1,data=z[tr],weights=w[tr])
  if(max(z$time_yr[tr])<3 || tail(c(1,g$surv[g$time<=3]),1)<.05) return(c(NA_real_,NA_real_))
  t=z$time_yr[va];e=z$event[va]; ww=w[va]
  if(!any(e==1 & t<=3)) return(c(0,0))
  G=c(1,g$surv)[1+findInterval(t,g$time,left.open=TRUE)]
  wins=den=0
  for(k in which(e==1 & t<=3)) {
    after=which(t>t[k])
    pair=ww[k]*ww[after]/G[k]^2
    wins=wins+sum(pair*((lp[k]>lp[after])+0.5*(lp[k]==lp[after])))
    den=den+sum(pair)
  }
  c(wins,den)
}
evaluate=function(z,design,w,uno=FALSE) {
  totals=matrix(0,nrow=length(design),ncol=if(uno)4 else 2)
  for(k in seq_len(5)) {
    tr=which(z$fold!=k & w>0);va=which(z$fold==k & w>0)
    if(!length(va)||!length(tr)||!any(z$event[tr]==1)) return(matrix(NA_real_,length(design),if(uno)2 else 1))
    for(i in seq_along(design)) {
      lp=model_lp(z,tr,va,design[[i]],w)
      totals[i,1:2]=totals[i,1:2]+pair_harrell(z$time_yr[va],z$event[va],lp,w[va])
      if(uno) totals[i,3:4]=totals[i,3:4]+pair_uno(z,tr,va,lp,w)
    }
  }
  if(uno) return(cbind(totals[,1]/totals[,2],totals[,3]/totals[,4]))
  matrix(totals[,1]/totals[,2],ncol=1)
}
boot_delta=function(z,design,w,survey=FALSE) {
  if(survey) {
    z[,unit:=paste(stratum,psu,sep="/")]
    domain_units=unique(z[,.(unit,stratum)])
    # Preserve the historical replicate order when every design PSU is in the domain.
    u=rbind(domain_units,survey_frame[!domain_units,on="unit"])
    groups=split(seq_len(nrow(u)),u$stratum)
    if(any(lengths(groups)<2)) return(list(ok=FALSE,reason="singleton survey PSU within a sampling-frame stratum"))
  } else {
    z[,unit:=family]
    u=unique(z[,.(unit,fold)]);groups=split(seq_len(nrow(u)),u$fold)
  }
  ix=match(z$unit,u$unit); vals=rep(NA_real_,B)
  set.seed(734102)
  for(b in seq_len(B)) {
    multiplicity=numeric(nrow(u))
    for(v in groups) {
      m=length(v); draws=if(survey)m-1L else m
      multiplicity[v]=tabulate(sample.int(m,draws,replace=TRUE),nbins=m)*(m/draws)
    }
    vals[b]=tryCatch({a=evaluate(z,design[1:2],w*multiplicity[ix]); a[1,1]-a[2,1]},error=function(e) NA_real_)
  }
  list(ok=all(is.finite(vals)),successful=sum(is.finite(vals)),requested=B,vals=vals)
}
one=function(label,clock,where=d,clin="clinical",extra=character(),survey=FALSE,boot=FALSE) {
  columns=c(clin,clock,"age10","sex","time_yr","event",extra,if(survey)c("weight","stratum","psu"))
  if(!all(columns%in%names(where))) {add_avail(label,0,paste("missing columns",paste(setdiff(columns,names(where)),collapse=",")));return()}
  z=copy(where[complete.cases(where[,..columns])])
  if(survey) z=z[weight>0 & is.finite(weight)]
  if(nrow(z)<30 || sum(z$event)<20 || length(unique(z$fold))<5) {add_avail(label,nrow(z),"insufficient paired deaths or household folds");return()}
  z[,a:=get(clin)];z[,b:=get(clock)]
  w=if(survey)z$weight/mean(z$weight) else rep(1,nrow(z))
  base=c("age10","sex",extra)
  design=list(clinical=c(base,"a"),dnam=c(base,"b"),joint=c(base,"a","b"))
  calc=tryCatch(evaluate(z,design,w,TRUE),error=function(e)e)
  if(inherits(calc,"error") || any(!is.finite(calc[,1]))) {
    reason=if(inherits(calc,"error")) conditionMessage(calc) else "no estimable out-of-fold Harrell C"
    add_avail(paste(label,clock),nrow(z),reason)
    return()
  }
  row=list(analysis=label,clock=clock,n=nrow(z),families=uniqueN(z$family),deaths=sum(z$event),
           deaths_3y=sum(z$event==1 & z$time_yr<=3),C_clinical=calc[1,1],C_dnam=calc[2,1],C_joint=calc[3,1],
           delta=calc[1,1]-calc[2,1],Uno3_clinical=calc[1,2],Uno3_dnam=calc[2,2],Uno3_joint=calc[3,2])
  if(boot) {
    s=boot_delta(z,design,w,survey)
    row$requested=B;row$successful=if(is.null(s$successful)) 0L else s$successful
    if(s$ok) {
      ci=quantile(s$vals,c(.025,.975),names=FALSE)
      row$lower=ci[1];row$upper=ci[2]
      if(label=="primary" && clock=="clock_GrimAge2") row$H1=if(ci[1]>-.010)"supported" else "not_supported"
    } else add_avail(label,nrow(z),if(is.null(s$reason)) "bootstrap has unestimable repetitions; interval withheld" else s$reason)
  }
  point[[length(point)+1]] <<- row
  if(label%in%c("primary","other_clock","all_clocks_complete")) {
    z[,a:=as.numeric(scale(a))];z[,b:=as.numeric(scale(b))]
    fits=lapply(design,function(co) coxph(reformulate(co,response="Surv(time_yr,event)"),data=z,cluster=family,ties="breslow"))
    for(m in names(fits)) {
      fit=fits[[m]]
      for(term in intersect(c("a","b"),names(coef(fit)))) {
        bb=coef(fit)[term];se=sqrt(vcov(fit)[term,term])
        effects[[length(effects)+1]] <<- list(analysis=label,clock=clock,model=m,term=term,HR=unname(exp(bb)),
                                                lower=unname(exp(bb-1.96*se)),upper=unname(exp(bb+1.96*se)),p=unname(2*pnorm(-abs(bb/se))))
      }
    }
    p=sapply(fits[1:2],function(f)pchisq(2*(fits$joint$loglik[2]-f$loglik[2]),1,lower.tail=FALSE))
    tests[[length(tests)+1]] <<- list(analysis=label,clock=clock,p_add_dnam=p[1],p_add_clinical=p[2],
                                       H2=if(label=="primary") all(p<.01) else NA)
  }
}
primary="clock_GrimAge2"
if(primary%in%clocks) {
  one("primary",primary,boot=TRUE)
  z=copy(d[time_yr>1]); z[,time_yr:=time_yr-1];one("one_year_landmark",primary,z)
  for(cap in c(80,85)) one(paste0("age_cap_",cap),primary,clin=paste0("clinical_cap",cap))
  for(field in c("time_lower_yr","time_upper_yr")) if(field%in%names(d)) {
    j=copy(d[get(field)>0 & is.finite(get(field))]); j[,time_yr:=get(field)]; one(field,primary,j)
  }
  cell=sort(grep("^cell_",names(d),value=TRUE))
  if(length(cell)>=2) one("cell_composition",primary,extra=head(cell,-1)) else add_avail("cell_composition",0,"cell fraction set unavailable")
  one("survey_design",primary,survey=TRUE,boot=TRUE)
} else add_avail("H1/H2",0,"GrimAge2 unavailable; no substituted primary clock")
for(clock in setdiff(clocks,primary)) one("other_clock",clock)
cc=d[complete.cases(d[,..clocks])]
for(clock in clocks) one("all_clocks_complete",clock,cc)
org=grep("^organ_",names(d),value=TRUE)
org=org[colSums(is.finite(as.matrix(d[,..org])))>=30]
for(cl in clocks) {
  for(panel in c("clinical",org)) {
    z=d[complete.cases(d[,c(cl,panel,"age","sex"),with=FALSE])]
    if(nrow(z)>=30) corr[[length(corr)+1]]=list(clock=cl,organ=panel,n=nrow(z),r=cor(resid(lm(z[[cl]]~age+sex,z)),resid(lm(z[[panel]]~age+sex,z))))
  }
  z=d[complete.cases(d[,c(cl,org,"age","sex"),with=FALSE])]
  if(nrow(z)>=30 && length(org)) {
    reg=lm(reformulate(c("age","sex",org),response=cl),data=z)
    # Household-clustered OLS sandwich; jointly adjusted for every available organ.
    X=model.matrix(reg); U=rowsum(X*resid(reg),z$family);G=nrow(U);N=nrow(X);P=ncol(X)
    if(qr(X)$rank==P && G>1 && N>P) {
      bread=chol2inv(chol(crossprod(X)))
      se=setNames(sqrt(diag(bread%*%crossprod(U)%*%bread)*G/(G-1)*(N-1)/(N-P)),colnames(X))
      for(term in org) organreg[[length(organreg)+1]]=list(clock=cl,organ=term,n=N,estimate=coef(reg)[term],
                lower=coef(reg)[term]-qt(.975,G-1)*se[term],upper=coef(reg)[term]+qt(.975,G-1)*se[term],
                p=2*pt(-abs(coef(reg)[term]/se[term]),G-1))
    }
  }
}
for(clock in clocks) add_avail(clock,sum(is.finite(d[[clock]])),paste("missing",sum(!is.finite(d[[clock]]))))
tabs=list(discrimination=point,HR=effects,LRT=tests,availability=avail,partial_correlation=corr,organ_regression=organreg)
for(name in names(tabs)) {
  tab=rbindlist(tabs[[name]],fill=TRUE)
  if("p"%in%names(tab)) tab[,q_BH:=p.adjust(p,"BH")]
  if(name=="LRT" && nrow(tab)) tab[,c("q_add_dnam_BH","q_add_clinical_BH"):=.(p.adjust(p_add_dnam,"BH"),p.adjust(p_add_clinical,"BH")),by=analysis]
  if(nrow(tab)) fwrite(tab,file.path(out,paste0(name,".csv")))
}
write_json(list(mode=arg[3],source_rows=source_rows,eligible=nrow(d),families=uniqueN(d$family),n_folds=5,
           household_split_violations=d[,uniqueN(fold),by=family][V1!=1,.N],bootstrap_refits=B,
           primary_complete=any(vapply(point,function(x)identical(x$analysis,"primary") &&
             identical(x$successful,B) && is.finite(x$lower) && is.finite(x$upper),FALSE)) &&
             any(vapply(tests,function(x)identical(x$analysis,"primary") &&
               !is.na(x$H2),FALSE)),
           input_hash=digest::digest(file=arg[1],algo="sha256"),survival=as.character(packageVersion("survival"))),
           file.path(out,"run_status.json"),pretty=TRUE,auto_unbox=TRUE)
