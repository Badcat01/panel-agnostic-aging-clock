suppressPackageStartupMessages({library(data.table);library(survival);library(nlme)})
out=commandArgs(trailingOnly=TRUE)[1]
rows=list(); mixed=list()
dx=c('hibpe','diabe','cancre','lunge','hearte','stroke','arthre','psyche')
for(w0 in c(2,4,6)){
  v=as.data.table(read.csv(gzfile(file.path(out,paste0('elsa_visits_w',w0,'_to_w9.csv.gz'))),colClasses=c(ID='character')))
  v=v[complete.cases(age,sex,education,full_z,time)];v[,education:=factor(education)];v[,age10:=(age-60)/10]
  setorder(v,ID,wave)
  for(ep in c('adl','iadl','reported_dementia',dx,'new_chronic','multimorbidity')){
    eligible=v[wave==w0&get(ep)==0,ID]
    d=v[ID%chin%eligible&wave>w0&is.finite(get(ep))&time>0]
    if(nrow(d)==0)next
    # Within-person interval from previous observed negative, allowing missing waves.
    d[,prior_time:=shift(time,fill=0),by=ID]
    d[,prior_positive:=shift(cummax(get(ep)),fill=0),by=ID]
    d=d[prior_positive==0&time>prior_time];d[,y:=get(ep)]
    if(sum(d$y)<20)next
    model=glm(y~factor(wave)+log(time-prior_time)+age10+I(age10^2)+sex+education+full_z,data=d,family=binomial())
    score_cluster=rowsum(model.matrix(model)*(d$y-fitted(model)),d$ID,reorder=FALSE)
    bread=vcov(model)
    robust=bread%*%crossprod(score_cluster)%*%bread
    b=coef(model)[['full_z']];se=sqrt(robust['full_z','full_z'])
    rows[[length(rows)+1]]=list(baseline_wave=w0,endpoint=ep,n=uniqueN(d$ID),observed_intervals=nrow(d),events=sum(d$y),
                       OR=exp(b),lo=exp(b-1.96*se),hi=exp(b+1.96*se),p=2*pnorm(-abs(b/se)))
  }
  for(ep in c('recall','fluency')){
    d=v[is.finite(get(ep))]
    u=d[,.(ok=any(wave==w0)&any(wave>w0)),by=ID][ok==TRUE,ID]
    d=d[ID%chin%u];if(nrow(d)<100)next
    d[,y:=get(ep)/sd(d[wave==w0,get(ep)])]
    fit=function(opt)lme(y~time*(age10+I(age10^2)+sex+education+full_z),random=~1|ID,data=d,
       control=lmeControl(opt=opt,msMaxIter=250))
    f=tryCatch(fit('nlminb'),error=function(e)fit('optim'))
    c=summary(f)$tTable['time:full_z',]
    mixed[[length(mixed)+1]]=list(baseline_wave=w0,endpoint=ep,n=uniqueN(d$ID),observations=nrow(d),beta_SD_year=c[1],lo=c[1]-1.96*c[2],hi=c[1]+1.96*c[2],p=c[5])
  }
}
a=rbindlist(rows);a[,q_per_landmark:=p.adjust(p,'BH'),by=baseline_wave]
b=rbindlist(mixed);b[,q:=p.adjust(p,'BH')]
fwrite(a,file.path(out,'elsa_observed_incidence.csv'));fwrite(b,file.path(out,'elsa_observed_cognition.csv'))
