# Mixed coding systems are mapped separately, then conditions are combined before severity hierarchy and weighting.
Sys.setlocale("LC_CTYPE", ".UTF-8")
.libPaths(c("R_lib", .libPaths()))
library(comorbidity)
library(data.table)
setDTthreads(3)
stopifnot(as.character(packageVersion("comorbidity"))=="1.1.0")
codes=unique(fread("cache/prior_codes.csv.gz", select=c("subject_id","icd_code","icd_version"),
                  colClasses=c(icd_code="character")))
stopifnot(all(codes$icd_version %in% c(9L,10L)), !anyNA(codes), all(nchar(codes$icd_code)>0))
key=unique(codes[,.(icd_code,icd_version)])
key[,code_id:=.I]
full=unique(merge(codes,key,by=c("icd_code","icd_version"))[,.(subject_id,code_id)])
build=function(kind,weight) {
  blocks=lapply(c(9L,10L),function(v) {
    part=key[icd_version==v]
    out=comorbidity(as.data.frame(part),id="code_id",code="icd_code",map=paste0(kind,"_icd",v,"_quan"),
                    assign0=FALSE, labelled=FALSE)
    as.data.table(out)
  })
  maps=rbindlist(blocks,use.names=TRUE)
  domain=merge(full,maps,by="code_id")[,!"code_id"]
  combined=domain[,lapply(.SD,max),by=subject_id]
  summ=as.data.frame(combined)
  class(summ)=c("comorbidity","data.frame")
  attr(summ,"map")=paste0(kind,"_icd10_quan")
  res=data.frame(subject_id=summ$subject_id,value=as.numeric(score(summ,weights=weight,assign0=TRUE)))
  names(res)[2]=kind;res
}
res=merge(build("charlson","charlson"),build("elixhauser","vw"),by="subject_id")
stopifnot(nrow(res)==uniqueN(codes$subject_id), !anyNA(res))
fwrite(res,"cache/prior_scores.csv.gz")
print(summary(res[,c("charlson","elixhauser")]))

