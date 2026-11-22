# Same-sample healthspan outcomes. Inputs are elapsed time from 2016 VBS and disjoint incident/death codes.
Sys.setlocale("LC_CTYPE",".UTF-8")
suppressPackageStartupMessages({library(survival);library(data.table);library(jsonlite)})
arg=commandArgs(trailingOnly=TRUE);stopifnot(length(arg)==2)
s=fromJSON(file.path(arg[2],"run_status.json"))
stopifnot(s$input_hash==digest::digest(file=arg[1],algo="sha256"))
d=fread(arg[1],colClasses=c(ID="character"))
d=d[qc_pass==1 & age>=50 & time_yr>0 & is.finite(clinical) & is.finite(sex) & !is.na(event)]
d[,fam:=sub("-.*","",ID)]
clocks=grep("^clock_",names(d),value=TRUE)
rows=list();skip=list()
failure=function(clock,ep,reason,n=0) skip[[length(skip)+1]] <<- list(clock=clock,endpoint=ep,n=n,reason=reason)
add=function(clock,ep,m,z,ev,term,b,se) {
  rows[[length(rows)+1]] <<- list(clock=clock,endpoint=ep,model=m,n=nrow(z),households=uniqueN(z$fam),
                              events=ev,term=term,ratio=unname(exp(b)),lower=unname(exp(b-1.96*se)),
                              upper=unname(exp(b+1.96*se)),p=unname(2*pnorm(-abs(b/se))))
}
fit_endpoint=function(clock,ep) {
    count=ep=="new_conditions"
    cols=if(count) c("new_conditions","years_at_new_conditions") else paste0(c("ti_","st_"),ep)
    need=c(cols,clock,"education")
    if(!all(need%in%names(d))) {failure(clock,ep,"missing mapped secondary data or education");return()}
    z=copy(d[complete.cases(d[,..need])])
    if(count) {
      z=z[new_conditions>=0 & years_at_new_conditions>0]
      stopifnot(all(z$new_conditions==floor(z$new_conditions)))
      if(sum(z$new_conditions)<20) {failure(clock,ep,"insufficient new conditions",nrow(z));return()}
    } else {
      z[,t:=get(cols[1])];z[,status:=get(cols[2])];z=z[t>0]
      stopifnot(all(z$status%in%0:2))
      if(sum(z$status==1)<20) {failure(clock,ep,"fewer than 20 incident cases",nrow(z));return()}
    }
    z[,a:=as.numeric(scale(clinical))];z[,b:=as.numeric(scale(get(clock)))]
    z[,education:=factor(education)]
    if(any(!is.finite(z$a) | !is.finite(z$b)) || nlevels(z$education)<2 || uniqueN(z$fam)<2) {
      failure(clock,ep,"nonestimable covariate or insufficient independent households",nrow(z))
      return()
    }
    predictors=model.matrix(~age+sex+education+a+b,z)
    if(qr(predictors)$rank<ncol(predictors)) {
      failure(clock,ep,"rank-deficient specified covariates",nrow(z))
      return()
    }
    if(count) {
      fit=glm(new_conditions~age+sex+education+a+b+offset(log(years_at_new_conditions)),data=z,family=poisson())
      X=model.matrix(fit); mu=fitted(fit)
      bread=chol2inv(chol(crossprod(X,X*as.numeric(mu))))
      U=rowsum(X*(z$new_conditions-mu),z$fam);g=nrow(U)
      se=setNames(sqrt(diag(bread%*%crossprod(U)%*%bread)*g/(g-1)),colnames(X))
      for(term in c("a","b"))add(clock,ep,"new chronic count, Poisson rate",z,sum(z$new_conditions),term,coef(fit)[term],se[term])
      return()
    }
    z[,outcome:=factor(status,levels=0:2,labels=c("censored","incident","death"))]
    cs=coxph(Surv(t,status==1)~age+sex+education+a+b,data=z,cluster=fam,ties="breslow")
    fg=finegray(Surv(t,outcome)~age+sex+education+a+b+fam,data=z,etype="incident")
    subdist=coxph(Surv(fgstart,fgstop,fgstatus)~age+sex+education+a+b,data=fg,
                  weights=fgwt,cluster=fam,ties="breslow")
    for(m in c("cause_specific","fine_gray")) {
      model=if(m=="cause_specific") cs else subdist
      for(term in c("a","b")) add(clock,ep,m,z,sum(z$status==1),term,coef(model)[term],sqrt(vcov(model)[term,term]))
    }
}
for(clock in clocks) for(ep in c("adl","lw","new_conditions")) fit_endpoint(clock,ep)
if(length(rows)){out=rbindlist(rows);out[,q_BH:=p.adjust(p,"BH"),by=model];fwrite(out,file.path(arg[2],"healthspan.csv"))}
if(length(skip))fwrite(rbindlist(skip),file.path(arg[2],"healthspan_unavailable.csv"))
