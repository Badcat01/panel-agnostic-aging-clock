suppressPackageStartupMessages({library(survival);library(nlme);library(data.table)})
out=commandArgs(trailingOnly=TRUE)[1]
openfile=function(s) as.data.table(read.csv(gzfile(file.path(out,s)),colClasses=c(ID='character')))
events=list();mixed=list();skips=list();org=list();comparison=list()
adjs='age10+I(age10^2)+sex+education'
dx=c('hibpe','dyslipe','diabe','cancre','lunge','livere','hearte','stroke','kidneye','digeste','psyche','memrye','arthre','asthmae')
for(w0 in c(1,3)){
  pre=paste0('charls_extended_w',w0)
  b=openfile(paste0(pre,'_base.csv.gz'));v=openfile(paste0(pre,'_visits.csv.gz'))
  stopifnot(!anyDuplicated(b$ID),!anyDuplicated(v[,.(ID,wave)]))
  b[,age10:=(age-60)/10];b[,education:=factor(education)]
  v=merge(v,b,by='ID')
  if(w0==1)b=b[!is.na(fi0)]  # align primary roster for extending the survey window
  for(ep in c('adl','iadl',dx,'new_chronic','multimorbidity')){
    u=v[wave==w0 & get(ep)==0,ID]
    z=v[ID%chin%u & wave>w0 & time>0 & time<=horizon & is.finite(get(ep)) & (death==0|time<=death_upper+.003)]
    o=z[,.(pos=if(any(get(ep)==1))min(time[get(ep)==1]) else Inf,last=max(time)),by=ID]
    d=merge(b[ID%chin%u],o,by='ID',all.x=TRUE)
    d[,status:=ifelse(is.finite(pos),1,ifelse(death==1&death_or_censor<=horizon,2,0))]
    d[,t:=ifelse(status==1,pos,ifelse(status==2,death_or_censor,last))]
    d=d[t>0&complete.cases(age10,education,sex,full_z)]
    if(sum(d$status==1)<20){skips[[length(skips)+1]]=list(wave=w0,endpoint=ep,n=nrow(d));next}
    form=as.formula(paste('Surv(t,status==1)~',adjs,'+full_z'))
    f=coxph(form,data=d,ties='efron')
    ci=summary(f)$conf.int['full_z',]
    rec=list(baseline=w0,endpoint=ep,n=nrow(d),events=sum(d$status==1),competing_deaths=sum(d$status==2),
             hr=ci[1],lo=ci[3],hi=ci[4],p=summary(f)$coef['full_z',5])
    if(ep%in%c('adl','iadl','multimorbidity','new_chronic')){
       d[,type:=factor(status,0:2,c('censored','diagnosis','death'))]
       long=finegray(Surv(t,type)~age10+sex+education+full_z+ID,data=d,etype='diagnosis')
       fg=coxph(Surv(fgstart,fgstop,fgstatus)~age10+I(age10^2)+sex+education+full_z,
                data=long,weights=fgwt,cluster=ID,ties='breslow')
       ci=summary(fg)$conf.int['full_z',]
       rec=c(rec,list(fg=ci[1],fg_lo=ci[3],fg_hi=ci[4]))
    }
    events[[length(events)+1]]=rec
    if(ep%in%c('diabe','livere','kidneye','hearte','stroke','lunge','cancre')){
      groups=c('kidney','metabolic','immune_heme','cardiovascular')
      dd=d[complete.cases(d[,..groups])]
      fits=lapply(groups,function(g)coxph(as.formula(paste('Surv(t,status==1)~',adjs,'+full_z+',g)),data=dd))
      names(fits)=groups
      for(g in groups){
         f=fits[[g]];ci=summary(f)$conf.int[g,]
         org[[length(org)+1]]=list(wave=w0,endpoint=ep,group=g,n=nrow(dd),events=f$nevent,hr=ci[1],lo=ci[3],hi=ci[4],p=summary(f)$coef[g,5])
      }
      match=c(diabe='metabolic',kidneye='kidney',hearte='cardiovascular',stroke='cardiovascular')[ep]
      if(!is.na(match)){
        be=sapply(groups,function(g)coef(fits[[g]])[[g]])
        inf=sapply(groups,function(g)residuals(fits[[g]],type='dfbeta')[,length(coef(fits[[g]]))])
        other=setdiff(groups,match);j=which(groups==match)
        pval=sapply(other,function(g)pnorm((be[j]-be[g])/sqrt(sum((inf[,j]-inf[,g])^2)),lower.tail=FALSE))
        comparison[[length(comparison)+1]]=list(baseline=w0,endpoint=ep,matched=match,n=nrow(dd),events=sum(dd$status==1),
               HR=exp(be[j]),strongest_other=other[which.max(be[other])],p_match=max(pval))
      }
    }
  }
  for(ep in c('recall','cesd10','dxcount','grip','walk_seconds')){
    d=v[is.finite(time)&time>=0&time<=horizon&is.finite(get(ep))&complete.cases(age10,sex,education,full_z)&(death==0|time<=death_upper+.003)]
    eligible=d[,.(ok=any(wave==w0)&any(wave>w0)),by=ID][ok==TRUE,ID]
    d=d[ID%chin%eligible]
    if(nrow(d)<50){skips[[length(skips)+1]]=list(wave=w0,endpoint=ep,n=uniqueN(d$ID));next}
    sd0=sd(d[wave==w0,get(ep)]);d[,y:=get(ep)/sd0]
    model=function(opt)lme(y~time*(age10+I(age10^2)+sex+education+full_z),random=~1|ID,data=d,
                             control=lmeControl(opt=opt,msMaxIter=250))
    f=tryCatch(model('nlminb'),error=function(e)model('optim'))
    tb=summary(f)$tTable['time:full_z',]
    mixed[[length(mixed)+1]]=list(wave=w0,endpoint=ep,n=uniqueN(d$ID),n_observations=nrow(d),waves=paste(sort(unique(d$wave)),collapse=';'),
                         sd_change_per_year=tb[1],lo=tb[1]-1.96*tb[2],hi=tb[1]+1.96*tb[2],p=tb[5])
  }
}
v=rbindlist(events,fill=TRUE);v[,q_per_landmark:=p.adjust(p,'BH'),by=baseline];fwrite(v,file.path(out,'charls_extended_events.csv'))
m=rbindlist(mixed);m[,q_per_landmark:=p.adjust(p,'BH'),by=wave];fwrite(m,file.path(out,'charls_extended_mixed.csv'))
o=rbindlist(org);o[,q:=p.adjust(p,'BH')];fwrite(o,file.path(out,'charls_extended_organs.csv'))
oc=rbindlist(comparison);oc[,q:=p.adjust(p_match,'BH')];fwrite(oc,file.path(out,'charls_extended_diagonal.csv'))
if(length(skips))fwrite(rbindlist(skips),file.path(out,'charls_extended_skips.csv'))

