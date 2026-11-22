# Diagnostic on the original 3 ADL and 2 matched-disease rosters.
# Fixed before inspecting diagnostics: strata at 4 years, >=20 events in each
# interval, unadjusted core p<.05 triggers descriptive interval estimates, BH
# across 8 stated core-term tests. Global tests include all covariates.
suppressPackageStartupMessages({library(data.table);library(survival)})
out=commandArgs(trailingOnly=TRUE)[1]
gettab=function(file) as.data.table(read.csv(gzfile(file.path(out,file)),colClasses=c(ID='character')))
previous=fread(file.path(out,'out.csv'))
organs=fread(file.path(out,'org.csv'))
rows=list(); intervals=list(); covs=list()
setDTthreads(2)
for(pop in c('hrs','elsa','charls')){
  b=gettab(paste0(pop,'_baseline.csv.gz'));v=gettab(paste0(pop,'_visits.csv.gz'))
  stopifnot(!anyDuplicated(b$ID),!anyDuplicated(v[,.(ID,wave)]))
  b[,age10:=(age-60)/10];b[,education:=factor(education)]
  v=merge(v,b,by='ID')
  endpoints=switch(pop,hrs=c('adl','diabe'),elsa='adl',charls=c('adl','kidneye'))
  for(ep in endpoints){
    elig=v[wave==min(wave)&get(ep)==0,ID];w0=min(v$wave)
    obs=v[ID%chin%elig & wave>w0 & is.finite(time)&time>0&time<=horizon&is.finite(get(ep))&(death==0|time<=death_upper+.003)]
    events=obs[,.(positive=if(any(get(ep)==1))min(time[get(ep)==1]) else Inf,observed=max(time)),by=ID]
    d=merge(b[ID%chin%elig],events,by='ID',all.x=TRUE)
    d[,status:=ifelse(is.finite(positive),1L,ifelse(death==1&death_or_censor<=horizon,2L,0L))]
    d[,t:=ifelse(status==1,positive,ifelse(status==2,death_or_censor,observed))]
    d=d[t>0&complete.cases(age10,sex,education,fi0,full_z,ridge_lp)]
    main='full_z'; specs=c(demographic='age10+I(age10^2)+sex+education+full_z')
    if(ep=='adl'){
      specs=c(specs,demographic_FI='age10+I(age10^2)+sex+education+fi0+full_z')
      prev=previous[cohort==pop&endpoint==ep]
    } else {
      main=if(ep=='diabe')'metabolic' else 'kidney'
      gr='liver kidney metabolic immune_heme cardiovascular'
      gr=strsplit(gr,' ',fixed=TRUE)[[1]]; gr=gr[sapply(gr,function(s)sum(is.finite(d[[s]]))>100)]
      d=d[complete.cases(d[,..gr])]
      specs=c(demographic_full=paste('age10+I(age10^2)+sex+education+full_z+',main))
      prev=organs[cohort==pop&endpoint==ep&organ==main]
    }
    for(mod in names(specs)){
      rhs=specs[[mod]]
      form=as.formula(paste('Surv(t,status==1)~',rhs))
      fit=coxph(form,data=d,ties='efron',x=TRUE)
      core=switch(mod,demographic=prev$demog_est,demographic_FI=prev$fiadj_est,demographic_full=prev$est)
      stopifnot(nrow(prev)==1,nrow(d)==prev$n,fit$nevent==prev$events,abs(exp(coef(fit)[[main]])-core)<1e-8)
      z=cox.zph(fit,transform='km',terms=TRUE,global=TRUE)$table
      p=z[main,'p']
      e1=sum(d$status==1&d$t<=4);e2=sum(d$status==1&d$t>4)
      rows[[length(rows)+1]]=list(cohort=pop,endpoint=ep,spec=mod,core_term=main,n=nrow(d),events=fit$nevent,core_HR=core,
         chisq=z[main,'chisq'],df=z[main,'df'],core_p=p,global_p=z['GLOBAL','p'],events_le_4y=e1,events_gt_4y=e2)
      covs[[length(covs)+1]]=data.table(cohort=pop,endpoint=ep,spec=mod,term=rownames(z),p=z[,'p'])
      if(p<.05 && min(e1,e2)>=20){
        early=copy(d);early[,y:=as.integer(status==1&t<=4)];early[,duration:=pmin(t,4)]
        late=copy(d[t>4]);late[,y:=as.integer(status==1)];late[,duration:=t-4]
        for(period in c('early','late')){
          part=if(period=='early')early else late
          f=coxph(as.formula(paste('Surv(duration,y)~',rhs)),data=part,ties='efron')
          beta=coef(f)[[main]];se=sqrt(vcov(f)[main,main])
          intervals[[length(intervals)+1]]=list(cohort=pop,endpoint=ep,spec=mod,core_term=main,period=period,n=nrow(part),events=f$nevent,
                  hr=exp(beta),lo=exp(beta-1.96*se),hi=exp(beta+1.96*se))
        }
      }
    }
  }
}
r=rbindlist(rows);stopifnot(nrow(r)==8);r[,core_q:=p.adjust(core_p,'BH')];fwrite(r,file.path(out,'cox_ph_diagnostics.csv'))
fwrite(rbindlist(covs),file.path(out,'cox_ph_all_terms.csv'))
int=rbindlist(intervals)
if(nrow(int)==0)int=data.table(cohort=character(),endpoint=character(),spec=character(),core_term=character(),period=character(),
                               n=integer(),events=integer(),hr=numeric(),lo=numeric(),hi=numeric())
fwrite(int,file.path(out,'cox_ph_flagged_intervals.csv'))
