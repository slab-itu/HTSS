library(plyr)
library(stringr)
library(e1071)
library(xlsx)
library(lsa)
library(stringdist)
library(tm)
library(wordnet)
library(openNLP)
library(RWeka)
library(coreNLP)
Sys.setenv(WNHOME="D:/thesis stuff/wn3.1.dict.tar/wn3.1.dict")
#mydata=data.frame(stringsAsFactors = FALSE)
#mydata=data.frame(cite="", stringsAsFactors=FALSE)

setDict("D:/thesis stuff/wn3.1.dict.tar/wn3.1.dict/dict")
setwd("C:/Users/anam/Desktop/paper implementation/MeaningfulCitationsDataset")

#load up word polarity list and format it
data <- read.xlsx("preprocess_data.xlsx", 1, as.data.frame=TRUE, header=TRUE, stringsAsFactors=FALSE)
txt<- as.matrix(data$Abstract)

txt1=as.matrix(txt[427:450])



  
##################################remove unnecessary characters and split up by word 
  
  preprocess=function(sentence){
    
          sentence=str_replace_all(sentence, "\n", "  ")
          sentence <- gsub('[[:cntrl:]]', '', sentence)
          sentence <- gsub('\\d+', '', sentence)
          corpus=Corpus(VectorSource(sentence));
          corpus <- tm_map(corpus,content_transformer(tolower));
          corpus <- tm_map(corpus, content_transformer(removePunctuation))
          corpus <- tm_map(corpus, removeWords, stopwords("english"));
          
          dataframe <- data.frame(id=sapply(corpus, meta, "id"),
                                  text=unlist(lapply(sapply(corpus, '[', "content"),paste,collapse="\n")),
                                  stringsAsFactors=FALSE)
          
          sent <- as.String(dataframe$text)
          sent=paste(str_extract_all(sent, '\\w{4,}')[[1]], collapse=' ')
          return(sent)

}
  
############################## POS tagging  ################################


## Need sentence and word token annotations.

posTag=function(s){
  
            sent_token_annotator <- Maxent_Sent_Token_Annotator()
            word_token_annotator <- Maxent_Word_Token_Annotator()
            a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
            
            pos_tag_annotator <- Maxent_POS_Tag_Annotator()
            a3 <- annotate(s, pos_tag_annotator, a2)
            a3=as.data.frame(a3)
            temp=a3$features
            temp=temp[c(2:length(temp))]
            temp1=unlist(temp)
            temp1=as.data.frame(temp1)
            final=lapply(temp1, universalTagset)
            return(final)
}
  
#######################################################################

  mydata1=data.frame(cite="", stringsAsFactors=FALSE)
  for (j in 1:length(txt1)){

      #result=preprocess(txt[401]) 
      wordList <- str_split(txt1[j], '\\s+')
      words <- unlist(wordList)
      words=as.matrix(words)
      postagg=posTag(txt1[j]) 
  
      mat=as.matrix(cbind(words,postagg$temp1))
      mat=mat[mat[,2] == "NOUN" | mat[,2] =="ADJ" | mat[,2] =="VERB" | mat[,2] =="ADV" ,]
      str=as.character()

for ( k in 1:nrow(mat)){
  
  x=synonyms(mat[k,1],mat[k,2])
  y=paste0(x, collapse = " ")
  f=y
  str=paste(str, f, sep = " ")

         }

         #mydata=rbind(mydata,str)
         mydata1=rbind(mydata1,str)
         
        #return(str)
}

#result=apply(txt1, 1, posTag) 

write.xlsx(mydata1, file = "synonyms_Abstract6.xlsx")

  ########################################################################


