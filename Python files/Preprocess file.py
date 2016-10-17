# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:38:18 2016

@author: anam
"""

  #import pyexcel as pe
from textblob import Word
#import numpy as np
import pandas as pd    
from bs4 import BeautifulSoup 
import re       
#import nltk
import string
from nltk.corpus import stopwords # Import the stop word list
#from nltk.stem import WordNetLemmatizer as wnl
#from nltk.stem import PorterStemmer as port
#from stemming.porter2 import stem
#from sklearn.feature_extraction.text import CountVectorizer

xls_file = pd.ExcelFile('C:/Users/anam/Desktop/paper implementation/MeaningfulCitationsDataset/cue_words_new.xlsx')
# Load the xls file's Sheet1 as a dataframe
df = xls_file.parse('Sheet1')
txt=df.Using
#txt1=txt[1]

def review_to_words( raw_txt ):
    raw_txt=raw_txt.replace("\n", " ")
      # 1. Remove HTML
    review_text = BeautifulSoup(raw_txt).get_text() 
      # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
      # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split() 
      #set of alphabets
    alpha=list(string.ascii_lowercase)
    
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words+ alphabets to a set
    stops = set(stopwords.words("english")+alpha)                  
        
        # 5. Remove stop words and single alphabets
    meaningful_words = [w for w in words if not w in stops]
       # meaningful_words = [w for w in  meaningful_words if not w in alpha]   
    words_removedStops=" ".join( meaningful_words)
      
     # applying lemmatization for verbs 'v', nouns 'n' and adjective 'a'
    words_lemma=" ".join([Word(i).lemmatize('v') for i in words_removedStops.split()])
    words_lemma=" ".join([Word(i).lemmatize('a') for i in words_lemma.split()])
    words_lemma=" ".join([Word(i).lemmatize('n') for i in words_lemma.split()])
    words_lemma =" ".join( [w for w in  words_lemma.split() if not w in alpha])   
    #words_lemma=" ".join([wnl.lemmatize(i) for i in words_removedStops.split()])
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    words_lemma= " ".join([w for w in  words_lemma.split() if len(w)>3])
     
    return(words_lemma) 

#dat=df.iloc[:,1:4]
df1=pd.DataFrame()

file=[]
for c in range(0,len(txt)):
    
    clean_text=review_to_words(txt[c])
    file.append(clean_text)
    
df1['Using']=file

writer=pd.ExcelWriter('cueWords_preprocess_data2.xlsx', engine='xlsxwriter')
df1.to_excel(writer, encoding='utf-8')
writer.save()
    
    



      
  
     

 