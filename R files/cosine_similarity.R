library(plyr)
library(stringr)
library(e1071)
library(xlsx)
library(lsa)
library(stringdist)
library(tm)
library(wordnet)
library(openNLP)

setDict("D:/thesis stuff/wn3.1.dict.tar/wn3.1.dict/dict")
setwd("C:/Users/anam/Desktop/paper implementation/MeaningfulCitationsDataset")

#load up the files to get the similarity between them.

abstract <- read.xlsx("synonyms_Abstract1.xlsx", 1, as.data.frame=TRUE, header=TRUE, stringsAsFactors=FALSE)

text <- read.xlsx("synonyms_text1.xlsx", 1, as.data.frame=TRUE, header=TRUE, stringsAsFactors=FALSE)

abs<- abstract$cite
txt<- text$cite



res=vector(length=length(txt))

#function to calculate number of words in each category within a sentence

for (j in 1:length(txt)){

    #remove unnecessary characters and split up by word 
#     sentence=str_replace_all(txt[j], "\n", " ")
#     sentence <- gsub('[[:punct:]]', '', sentence)
#     sentence <- gsub('[[:cntrl:]]', '', sentence)
#     sentence <- gsub('\\d+', '', sentence)
#     sentence <- tolower(sentence)
    #sentence=stemDocument(sentence)
    #wordList <- str_split(sentence, '\\s+')
    #words <- unlist(wordList)
    
    #remove unnecessary characters and split up by word 
#     abs_s=str_replace_all(abs[j], "\n", " ")
#     abs_s <- gsub('[[:punct:]]', '', abs_s)
#     abs_s <- gsub('[[:cntrl:]]', '', abs_s)
#     abs_s <- gsub('\\d+', '', abs_s)
#     abs_s <- tolower(abs_s)
    #abs_s=stemDocument(abs_s)
    #wordList1 <- str_split(abs_s, '\\s+')
    #words1 <- unlist(wordList1)
    
    #build vector with matches between sentence and each category
    sen=rbind(txt[j], abs[j])
    corpus=Corpus(VectorSource(sen));
    #corpus <- tm_map(corpus, removeWords, stopwords("english"));
    #myCorpus=corpus
    #corpus=tm_map(corpus, stemDocument, language = "english")
    #corpus=tm_map(corpus, stemCompletion, dictionary=myCorpus)
    dtm <- TermDocumentMatrix(corpus)
      
    ##### cosine using tf
    tf <- dtm
    tf=as.matrix(tf)
    dot=tf[,1]*tf[,2]
    do=sum(dot)
    final=sqrt(sum((tf[,1])^2))*sqrt(sum((tf[,2])^2))
    res[j]=do/final
    
}



    write.csv(res, file = "Cosine_similarity.csv")







