# Cause-specific Cox, competing-risk sensitivity and repeated cognitive performance.
suppressPackageStartupMessages({library(survival);library(data.table);library(nlme)})
arg <- commandArgs(trailingOnly=TRUE)
folder <- normalizePath(arg[1])
setDTthreads(2)
out <- list()
org <- list()
contrast_rows <- list()
mixed <- list()
disc <- list()
biological <- list()
skips <- list()
extract <- function(f, term) {
  b <- unname(coef(f)[term]); se <- sqrt(vcov(f)[term,term])
  c(est=exp(b),lower=exp(b-1.96*se),upper=exp(b+1.96*se),p=2*pnorm(-abs(b/se)))
}
asrow <- function(x) as.list(x)
fitcox <- function(d, rhs) coxph(as.formula(paste('Surv(t,status==1) ~',rhs)),data=d,ties='efron', model=FALSE)
for (cohort in c('hrs','elsa','charls')) {
  b <- as.data.table(read.csv(gzfile(file.path(folder,paste0(cohort,'_baseline.csv.gz'))),colClasses=c(ID='character')))
  bio_path=file.path(folder,'..','models','fits','M7_bioage',paste0('external_',cohort,'.csv.gz'))
  bio <- as.data.table(read.csv(gzfile(bio_path),colClasses=c(ID='character')))
  stopifnot(!anyDuplicated(bio$ID),setequal(b$ID,bio$ID))
  b <- merge(b,bio[,.(ID,M7_KDM,M7_PhenoAge)],by='ID',sort=FALSE)
  l <- as.data.table(read.csv(gzfile(file.path(folder,paste0(cohort,'_visits.csv.gz'))),colClasses=c(ID='character')))
  setorder(l,ID,wave); stopifnot(!anyDuplicated(b$ID),!anyDuplicated(l[,.(ID,wave)]))
  w0 <- min(l$wave)
  b[,age10:=(age-60)/10]; b[,education:=factor(education)]
  l <- merge(l,b,by='ID',all.x=TRUE,sort=FALSE)
  # Each wave has its own interview time; administrative endpoint per baseline origin.
  stopifnot('horizon'%in%names(b))
  binary <- intersect(c('adl','iadl','frailty','multimorbidity','new_chronic','lw_self_dementia','reported_dementia',
                        'hibpe','diabe','cancre','lunge','hearte','stroke','arthre','psyche','dyslipe','livere','kidneye','digeste','asthmae','memrye'),names(l))
  for (endpoint in binary) {
    a <- l[wave==w0 & get(endpoint)==0, .(ID)]
    z <- l[ID%chin%a$ID & !is.na(time) & wave!=w0 & time>0 & time<=horizon & !is.na(get(endpoint)) &
             (death==0 | time<=death_upper+0.003)]
    visited <- z[,.(first_pos=if(any(get(endpoint)==1)) min(time[get(endpoint)==1]) else Inf,
                      last_obs=max(time)),by=ID]
    d <- merge(b[ID%chin%a$ID],visited,by='ID',all.x=TRUE)
    death_by_end <- d$death==1 & d$death_or_censor<=d$horizon
    d[,status:=ifelse(is.finite(first_pos),1,ifelse(death_by_end,2,0))]
    d[,t:=fifelse(status==1,first_pos,fifelse(status==2,death_or_censor,last_obs))]
    d<-d[!is.na(t)&t>0 & complete.cases(age10,sex,education,fi0,full_z,ridge_lp)]
    if(sum(d$status==1)<20) {
      skips[[length(skips)+1]] <- list(cohort=cohort,endpoint=endpoint,why='less_than_20_events',n=nrow(d),events=sum(d$status==1))
      next
    }
    rhs <- 'age10 + I(age10^2) + sex + education'
    full <- fitcox(d,paste(rhs,'+ full_z'))
    adjusted <- fitcox(d,paste(rhs,'+ fi0 + full_z'))
    vals <- extract(full,'full_z'); adj <- extract(adjusted,'full_z')
    rw <- c(list(cohort=cohort,endpoint=endpoint,n=nrow(d),events=sum(d$status==1),deaths_compete=sum(d$status==2),
                 base_free_n=nrow(a),median_followup=median(d$t)),
             asrow(setNames(vals,paste0('demog_',names(vals)))),asrow(setNames(adj,paste0('fiadj_',names(adj)))))
    if(endpoint%in%c('adl','iadl')){
      dx=d[is.finite(fi_without_adl_iadl0)]
      reduced=fitcox(dx,paste(rhs,'+ fi_without_adl_iadl0 + full_z'))
      ab=extract(reduced,'full_z')
      rw=c(rw,list(nonADL_FI_n=nrow(dx),nonADL_FI_events=sum(dx$status==1)),asrow(setNames(ab,paste0('nonADL_FI_',names(ab)))))
    }
    if(endpoint%in%c('adl','iadl','frailty','lw_self_dementia','reported_dementia','new_chronic','multimorbidity')){
      d[,type:=factor(status,0:2,c('cens','incident','death'))]
      long <- finegray(Surv(t,type) ~ age10 + sex + education + fi0 + full_z + ID,data=d,etype='incident')
      fgfit <- coxph(Surv(fgstart,fgstop,fgstatus)~age10+I(age10^2)+sex+education+fi0+full_z,
                     data=long,weights=fgwt,cluster=ID,ties='breslow')
      fg <- extract(fgfit,'full_z')
      rw <- c(rw,asrow(setNames(fg,paste0('fg_',names(fg)))))
      # Compare alternative risk markers on held-out patients. No target-specific tuning.
      set.seed(20261121)
      d[,fold:=sample(rep(1:5,length.out=.N)),by=status]
      rr <- c(demog=rhs,FI=paste(rhs,'+ fi0'),clock=paste(rhs,'+ full_z'),
              ridge=paste(rhs,'+ ridge_lp'),FI_clock=paste(rhs,'+ fi0 + full_z'))
      for (method in names(rr)) {
        sides <- c(0,0)
        for(k in 1:5){
          m <- fitcox(d[fold!=k],rr[[method]])
          dd <- d[fold==k]; pred <- predict(m,newdata=dd,type='lp')
          cs <- concordance(Surv(dd$t,dd$status==1)~pred,reverse=TRUE,timewt='n')
          counts <- cs$count
          sides <- sides+c(counts[1]+0.5*counts[3],sum(counts[1:3]))
        }
        disc[[length(disc)+1]] <- list(cohort=cohort,endpoint=endpoint,n=nrow(d),model=method,C=sides[1]/sides[2])
      }
      db=d[complete.cases(M7_KDM,M7_PhenoAge)]
      if(nrow(db)>100 && sum(db$status==1)>20){
        set.seed(20261121); db[,fold:=sample(rep(1:5,length.out=.N)),by=status]
        methods=c(FI='fi0',clock='full_z',ridge='ridge_lp',PhenoAge='M7_PhenoAge',KDM='M7_KDM')
        for(method in names(methods)){
          total=c(0,0)
          for(k in 1:5){
            mod=fitcox(db[fold!=k],paste(rhs,'+',methods[[method]]))
            dv=db[fold==k]; lp=predict(mod,newdata=dv,type='lp')
            ct=concordance(Surv(dv$t,dv$status==1)~lp,reverse=TRUE)$count
            total=total+c(ct[1]+.5*ct[3],sum(ct[1:3]))
          }
          biological[[length(biological)+1]]=list(cohort=cohort,endpoint=endpoint,n=nrow(db),events=sum(db$status==1),
                method=method,within_fold_concordance=total[1]/total[2])
        }
      }
    }
    out[[length(out)+1]] <- rw
    if(endpoint%in%c('diabe','kidneye','livere','hearte','stroke','lunge','cancre')){
      organs <- c('liver','kidney','metabolic','immune_heme','cardiovascular')
      active <- organs[sapply(organs,function(s)sum(is.finite(d[[s]]))>100)]
      dc <- d[complete.cases(d[,..active])]
      if(nrow(dc)>100 && sum(dc$status==1)>=20) {
        fits <- list()
        for(group in active){
          f <- fitcox(dc,paste(rhs,'+ full_z +',group))
          fits[[group]] <- f
          org[[length(org)+1]] <- c(list(cohort=cohort,endpoint=endpoint,organ=group,n=nrow(dc),
                                      events=sum(dc$status==1),compared=paste(active,collapse=';')),asrow(extract(f,group)))
        }
        match_group <- c(diabe='metabolic',kidneye='kidney',livere='liver',hearte='cardiovascular',stroke='cardiovascular')[endpoint]
        if(!is.na(match_group) && match_group%in%active) {
          be <- sapply(active,function(g) unname(coef(fits[[g]])[g]))
          inf <- sapply(active,function(g) residuals(fits[[g]],type='dfbeta')[,length(coef(fits[[g]]))])
          others <- setdiff(active,match_group); j <- match(match_group,active)
          pvals<-sapply(others,function(g){
             dg <- be[j]-be[g]; influence <- inf[,j]-inf[,g]
             # Paired, robust one-sided Wald test for the log-HR difference.
             pnorm(dg/sqrt(sum(influence^2)),lower.tail=FALSE)
          })
          contrast_rows[[length(contrast_rows)+1]] <- list(cohort=cohort,endpoint=endpoint,matched=match_group,
              compared=paste(active,collapse=';'),n=nrow(dc),events=sum(dc$status==1),
              matched_hr=exp(be[j]),strongest_other=others[which.max(be[others])],
              strongest_other_hr=exp(max(be[others])),target_minus_best_other_logHR=be[j]-max(be[others]),
              p_intersection_union=max(pvals))
        }
      }
    }
  }
  for (endpoint in intersect(c('recall','grip','walk_seconds','cesd10','fluency','fi','dxcount'),names(l))){
    d<-l[is.finite(get(endpoint)) & is.finite(time) & time<=horizon & (death==0 | time<=death_upper+.003) &
           complete.cases(age10,sex,education,fi0,full_z)]
    # Include a baseline value and at least one follow-up value per patient.
    eligible <- d[,.(ok=any(wave==w0)&any(wave>w0)),by=ID][ok==TRUE,ID]
    d<-d[ID%chin%eligible]
    if(length(eligible)<50) {skips[[length(skips)+1]] <- list(cohort=cohort,endpoint=endpoint,why='insufficient_repeated_observations',n=length(eligible),events=NA);next}
    sd0 <- sd(d[wave==w0,get(endpoint)])
    d[,y:=get(endpoint)/sd0]
    # Include independent baseline FI and its interaction with time except for FI trajectory itself.
    base <- 'age10 + I(age10^2) + sex + education + full_z'
    build <- function(optimizer) lme(as.formula(paste('y ~ time * (',base,')')),random=~1|ID,data=d,
                                     method='REML',control=lmeControl(opt=optimizer,msMaxIter=250))
    fit <- tryCatch(build('nlminb'),error=function(e) build('optim'))
    demo <- summary(fit)$tTable['time:full_z',]
    if(endpoint!='fi'){
      base <- paste(base,'+ fi0')
      fit <- tryCatch(build('nlminb'),error=function(e) build('optim'))
    }
    tb <- summary(fit)$tTable['time:full_z',]
    mixed[[length(mixed)+1]] <- list(cohort=cohort,endpoint=endpoint,n=length(eligible),observations=nrow(d),base_sd=sd0,
                  beta_sd_per_year=tb[1],lo=tb[1]-1.96*tb[2],hi=tb[1]+1.96*tb[2],p=tb[5],
                  beta_native_per_year=tb[1]*sd0,demog_beta=demo[1],demog_lo=demo[1]-1.96*demo[2],demog_hi=demo[1]+1.96*demo[2],demog_p=demo[5])
  }
  print(paste('finished',cohort))
  for (nm in c('out','disc','org','mixed','skips')) if(length(get(nm))) fwrite(rbindlist(get(nm),fill=TRUE),file.path(folder,paste0(nm,'.csv')))
}
qvals <- rbindlist(out,fill=TRUE); qvals[,fiadj_q:=p.adjust(fiadj_p,method='BH')]
qvals[,primary_ADL_cognitive_q:=NA_real_]
m <- rbindlist(mixed)
family_p <- c(qvals[endpoint=='adl',demog_p],m[endpoint=='recall',demog_p])
q <- p.adjust(family_p,'BH'); n1 <- sum(qvals$endpoint=='adl')
qvals[endpoint=='adl',primary_ADL_cognitive_q:=q[seq_len(n1)]]
m[,q_all_mixed:=p.adjust(p,'BH')];m[endpoint=='recall',primary_ADL_cognitive_q:=q[-seq_len(n1)]]
sensq <- p.adjust(c(qvals[endpoint=='adl',fiadj_p],m[endpoint=='recall',p]),'BH')
qvals[endpoint=='adl',FI_ADL_cognitive_q:=sensq[seq_len(n1)]]
m[endpoint=='recall',FI_ADL_cognitive_q:=sensq[-seq_len(n1)]]
fwrite(qvals,file.path(folder,'out.csv')); fwrite(m,file.path(folder,'mixed.csv'))
o <- rbindlist(org); o[,q_all:=p.adjust(p,'BH')];fwrite(o,file.path(folder,'org.csv'))
oc <- rbindlist(contrast_rows); oc[,q_diagonal:=p.adjust(p_intersection_union,'BH')];fwrite(oc,file.path(folder,'disease_diagonal.csv'))
fwrite(rbindlist(biological),file.path(folder,'biological_age_comparisons.csv'))

