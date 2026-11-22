suppressPackageStartupMessages({library(survival);library(data.table)})
folder=normalizePath(commandArgs(trailingOnly=TRUE)[1])
openfile=function(name) as.data.table(read.csv(gzfile(file.path(folder,name)),colClasses=c(ID='character')))
rows=list()
run=function(dd,cohort,ep,term,rhs){
  d=dd[complete.cases(dd[,c('t','status','age10','sex','education',term),with=FALSE]) & t>0]
  if(sum(d$status==1)<20)return(NULL)
  f=coxph(as.formula(paste('Surv(t,status==1)~',rhs)),data=d,x=TRUE)
  tb=summary(f)$coef[term,]; ci=summary(f)$conf.int[term,]
  list(cohort=cohort,endpoint=ep,n=nrow(f$x),events=f$nevent,term=term,hr=ci[1],lo=ci[3],hi=ci[4],p=tb[5])
}
for (cohort in c('charls','elsa')){
  w=if(cohort=='elsa')4 else 3
  prefix=if(cohort=='charls')'charls_w5' else 'elsa'
  b=openfile(paste0(prefix,'_baseline.csv.gz'))
  l=openfile(paste0(prefix,'_visits.csv.gz'))
  pace=openfile(paste0(cohort,'_',w,'_pace.csv.gz'))
  pp=merge(pace,b,by='ID')
  pp[,age10:=(landmark_age-60)/10];pp[,education:=factor(education)]
  # Original cohort-specific mortality follow-up; restrict entry to survivors with post-visit time.
  d=pp[,.(age10,sex,education,baseline_ba,pace,t=death_or_censor-time_yr,status=death)]
  rows[[length(rows)+1]]=run(d,cohort,'death_after_second_draw','pace','age10 + I(age10^2) + sex + education + baseline_ba + pace')
  # ADL-free at the second-wave core assessment and at every prior observed assessment.
  ok=l[wave<=w,.(free=any(wave==w & adl==0,na.rm=TRUE) & !any(adl==1,na.rm=TRUE)),by=ID][free==TRUE]
  dx=l[wave>w & !is.na(adl) & !is.na(time) & ID%chin%ok$ID]
  dx=merge(dx[,.(ID,time,adl)],pp,by='ID')[time>time_yr & (death==0 | time<=death_upper+.003)]
  ob=dx[,.(pos=if(any(adl==1))min(time[adl==1]) else Inf,last=max(time)),by=ID]
  d=merge(pp[ID%chin%ok$ID],ob,all.x=TRUE,by='ID')
  d[,status:=ifelse(is.finite(pos),1,ifelse(death==1 & death_or_censor<=horizon,2,0))]
  d[,t:=fifelse(status==1,pos,fifelse(status==2,death_or_censor,last))-time_yr]
  rows[[length(rows)+1]]=run(d,cohort,'ADL_after_second_draw','pace','age10 + I(age10^2) + sex + education + baseline_ba + pace')
  if(cohort=='charls') {
    b[,age10:=(age-60)/10];b[,education:=factor(education)]
    for(ep in c('adl','iadl')){
      riskids=l[wave==1 & get(ep)==0,ID]
      fu=merge(l[wave>1 & !is.na(time)&!is.na(get(ep)) & ID%chin%riskids],b,by='ID')[death==0 | time<=death_upper+.003]
      obs=fu[,.(pos=if(any(get(ep)==1))min(time[get(ep)==1]) else Inf,last=max(time)),by=ID]
      d=merge(b[ID%chin%riskids],obs,all.x=TRUE,by='ID')
      d[,status:=ifelse(is.finite(pos),1,ifelse(death==1 & death_or_censor<=horizon,2,0))]
      d[,t:=fifelse(status==1,pos,fifelse(status==2,death_or_censor,last))]
      d=d[!is.na(fi0)]
      for(extra in c(FALSE,TRUE)){
        rows[[length(rows)+1]]=run(d,'charls',paste0(ep,'_through_2020',if(extra)'_FI' else ''),'full_z',
                      paste('age10 + I(age10^2) + sex + education + full_z',if(extra)'+ fi0' else ''))
      }
    }
  }
}
res=rbindlist(rows,fill=TRUE);res[,q_row_family:=p.adjust(p,'BH')]
fwrite(res,file.path(folder,'extended_landmarks.csv'))

