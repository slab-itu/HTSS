
setwd("D:/thesis stuff/Final paper implementation/MeaningfulCitationsDataset")

#load up word polarity list and format it

data <- read.csv(file='normalized_test_reverse class.csv', header=T, stringsAsFactors=FALSE)
data= data[,1:14]


x= data[,1:13]
y= data[,14]

p_corr1= cor( x, y, method = "pearson")

s_corr1 = cor(x, y, method="spearman")
       
require(corrplot)

corrplot(s_corr, method= "number")
