# to show how to retrieve from corpus
##lapply(text[1:3], as.character)

download.folder = "C:/Users/anam/Desktop/download papers/3 class/cited by/"

setwd("C:/Users/anam/Desktop/download papers")

data <- read.csv(file='myData.csv', header=T, stringsAsFactors=FALSE)


for (i in 1:85){
  
  pdf.name <- paste(download.folder, data$Cited_by[i], '.pdf', sep='')
  download.file(data$Cited_by_URL[i], pdf.name, mode="wb")
}
 

