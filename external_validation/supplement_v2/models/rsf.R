suppressPackageStartupMessages(library(randomForestSRC))
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(data.table))
options(rf.cores = 2L, mc.cores = 2L)
Sys.setenv(OMP_NUM_THREADS=2)
args = commandArgs(trailingOnly=TRUE)
panels = fromJSON("panel_definitions.json", simplifyVector=FALSE)$panels
d = as.data.frame(fread("cache/dev.csv.gz"))
years = sort(unique(d$cycle))

readfold = function(idx, markers) {
  dat = d[idx,c("permth_exm","event","age",markers)]
  m = sapply(dat[,c("age",markers)], median, na.rm=TRUE)
  stopifnot(all(is.finite(m)))
  list(data=fill(dat,m), med=m)
}
fill = function(dat,m) {
  for (n in names(m)) dat[[n]][is.na(dat[[n]])] = m[n]
  dat
}
fitforest = function(f, node) {
  set.seed(20261121)
  rfsrc(Surv(permth_exm, event) ~ ., data=f$data, ntree=128, nodesize=node,
        nsplit=5, ntime=32, importance="none", forest=TRUE,
        perf.type="none", seed=-20261121, block.size=1)
}
risk = function(obj,f) {
  blocks=split(seq_len(nrow(f)), ceiling(seq_len(nrow(f))/2048))
  unlist(lapply(blocks,function(b) predict(obj,newdata=f[b,,drop=FALSE],perf.type="none")$predicted),use.names=FALSE)
}
choose = function(rows) {
  avg=aggregate(c ~ node, rows, mean)
  avg$node[order(-avg$c,-avg$node)][1]
}
for (name in args) {
  loc=paste0("fits/M4_rsf_",name)
  dir.create(loc,showWarnings=FALSE,recursive=TRUE)
  markers=unlist(panels[[name]])
  grid=c(24,64)
  allrows = NULL
  for (om in c(combn(years,2,simplify=FALSE),as.list(years))) {
    id=paste(om,collapse="_")
    filename=file.path(loc,paste0("cv_",id,".csv"))
    if (!file.exists(filename)) {
      ft=readfold(!d$cycle %in% om,markers)
      result=NULL
      for (node in grid) {
        fit=fitforest(ft,node)
        for (yr in om) {
          v=d[d$cycle==yr,]
          vv=fill(v[,c("age",markers),drop=FALSE],ft$med)
          pred=risk(fit,vv)
          cc=concordance(Surv(permth_exm,event)~pred, data=v, reverse=TRUE)$concordance
          result=rbind(result,data.frame(omitted=id,year=yr,node=node,c=cc))
        }
      }
      fwrite(result,paste0(filename,".tmp"))
      stopifnot(file.rename(paste0(filename,".tmp"),filename))
    }
    allrows=rbind(allrows,as.data.frame(fread(filename,colClasses=c(omitted="character"))))
    cat(name,id,"\n")
  }
  single=!grepl("_",allrows$omitted)
  for (yr in years) {
    inner=allrows[!single & grepl(as.character(yr),allrows$omitted) & allrows$year!=yr,]
    outer=allrows[single & allrows$year==yr,]
    cv=outer[outer$node==choose(inner),]
    fwrite(cv,file.path(loc,paste0("nested_",yr,".csv")))
  }
  modelpath=file.path(loc,"final.rds")
  if (!file.exists(modelpath)) {
    node=choose(allrows[single,])
    ft=readfold(seq_len(nrow(d)),markers)
    fit=fitforest(ft,node)
    saveRDS(list(fit=fit,med=ft$med,node=node,version=as.character(packageVersion("randomForestSRC"))),
            paste0(modelpath,".tmp"), compress=FALSE)
    stopifnot(file.rename(paste0(modelpath,".tmp"),modelpath))
  }
  final=readRDS(modelpath)
  for(kind in c("sim","external","primary")) {
    path=file.path("cache",paste0(kind,"_",name,".csv.gz"))
    if(!file.exists(path)) next
    df=as.data.frame(fread(path,colClasses=c(ID="character")))
    outpath=file.path(loc,paste0(kind,".csv.gz"))
    stamp=file.path(loc,paste0(kind,"_input_hashes.csv"))
    hashes=unname(tools::md5sum(c(path,modelpath)))
    if (file.exists(outpath) && file.exists(stamp) &&
        identical(as.character(fread(stamp)$hash),hashes)) next
    riskv=risk(final$fit,fill(df[,c("age",markers),drop=FALSE],final$med))
    stopifnot(length(riskv)==nrow(df),all(is.finite(riskv)))
    fwrite(data.frame(ID=df$ID,M4_RSF=riskv),outpath)
    fwrite(data.frame(hash=hashes),stamp)
  }
}
