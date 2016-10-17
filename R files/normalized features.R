library(e1071)

setwd("D:/thesis stuff/Final paper implementation/MeaningfulCitationsDataset")

#load up word polarity list and format it

data <- read.csv(file='testing.csv', header=T, stringsAsFactors=FALSE)

# scaling 

feat= scale(data[,1:13], center = TRUE, scale = TRUE)

mydata=cbind(feat,data[,14])
write.csv(mydata, file = "normalized_test.csv")

