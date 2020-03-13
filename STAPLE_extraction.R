
# install.packages("stapler")
# install.packages("OpenImageR")
# install.packages("imager")
# install.packages("RNifti")
# install.packages("png")
# install.packages("ijtiff")

# library(ijtiff)
library(stringr)
library(png)
library(OpenImageR)
library(stapler)
library(RNifti)


setwd("C:\\Users\\Philo\\Documents\\3A -- MVA\\DL for medical imaging\\retine\\dlmi-project\\")


DATABASES=c("stare","chasedb1","aria")
DB_NAME=DATABASES[2]
path=paste0('data\\',DB_NAME)

if (DB_NAME != "aria"){
annot1=NULL
annot2=NULL
for (im in list.files(paste0(path,"\\annotation 1"))){
  print(im)
  if (im != "desktop.ini"){
    im_file1=paste0(path,"\\annotation 1\\",im)
    annot1=c(annot1,readImage(im_file1))
    print(max(readImage(im_file1)))
  }
}

for (im in list.files(paste0(path,"\\annotation 2"))){
  print(im)
  if (im != "desktop.ini"){
  im_file2=paste0(path,"\\annotation 2\\",im)
  annot2=c(annot2,readImage(im_file2))}
}

dim_im=dim(readImage(im_file2))
print(dim_im)

# list_imname=substr(list.files(paste0(path,"\\annotation 1")),1,6) #stare
# list_imname=substr(list.files(paste0(path,"\\annotation 1")),1,9) #chasedb1
# list_imname=list_imname[2:length(list_imname)]
list_imname=substr(list.files(paste0(path,"\\annotation 1")),1,11) 
list_imname=list_imname[2:(length(list_imname))]

annot=rbind(annot1,annot2)
dim(annot)
st=staple_bin_mat(annot) 
st$label
unique(st$label)


list_output=NULL
for (i in 0:(length(list_imname)-1)){
  print(i)
  print(list_imname[i+1])
  im=st$label[((i*dim_im[1]*dim_im[2])+1):(((i+1)*dim_im[1]*dim_im[2]))]
  im=matrix(im*1)
  dim(im)=dim_im
  print(dim(im))
  writePNG(im,paste0(path,"\\STAPLE\\",list_imname[i+1],".png"))
  # print(image(im))
}

saveRDS(st, paste0(path,"STAPLE_object.RDS") )
}


# suffix="aria_a"
suffix="aria_c"
# suffix="aria_d"

if (DB_NAME=="aria"){
  annot1=NULL
  annot2=NULL
  for (im in list.files(paste0(path,"\\annotation 1"))){
    if (substr(im,1,6)==suffix){ 
      print(im)
      im_file1=paste0(path,"\\annotation 1\\",im)
      annot1=c(annot1,readImage(im_file1))}
    }
  
  for (im in list.files(paste0(path,"\\annotation 2"))){
    if (substr(im,1,6)==suffix){
      print(im)
      im_file2=paste0(path,"\\annotation 2\\",im)
      annot2=c(annot2,readImage(im_file2))}
  }
  
  dim_im=dim(readImage(im_file2))
  print(dim_im)
  
  # list_imname=substr(list.files(paste0(path,"\\annotation 1")),1,6) #stare
  # list_imname=substr(list.files(paste0(path,"\\annotation 1")),1,9) #chasedb1
  # list_imname=list_imname[2:length(list_imname)]
  list_imname=list.files(paste0(path,"\\annotation 1"))
  list_imname=list_imname[substr(list_imname,1,6)==suffix]
  list_imname=str_sub(list_imname, end=-5)
  list_imname=sapply(list_imname,function(x){gsub("_BDP","",x)})
  length(unique(list_imname))
  annot=rbind(annot1,annot2)

  dim(annot)
  st=staple_bin_mat(annot) 
  st$label
  unique(st$label)
  length(st$label)/(dim_im[1]*dim_im[2])
  length(annot1)/(dim_im[1]*dim_im[2])
  
  list_output=NULL
  for (i in 0:(length(list_imname)-1)){
    print(i)
    print(list_imname[i+1])
    im=st$label[((i*dim_im[1]*dim_im[2])+1):(((i+1)*dim_im[1]*dim_im[2]))]
    im=matrix(im*1)
    dim(im)=dim_im
    print(dim(im))
    writePNG(im,paste0(path,"\\STAPLE\\",list_imname[i+1],".png"))
    # print(image(im))
  }
  
  saveRDS(st, paste0(path,"STAPLE_object",suffix,".RDS") )
}

# st_open=readRDS(paste0(path,"STAPLE_object.RDS"))
