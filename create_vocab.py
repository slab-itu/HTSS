# import spacy
import nltk
import pandas as pd

import data_util.config as config

# nlp = spacy.load("en_core_web_sm")
print("Crating dictionary please wait ... .")

data_file = pd.read_csv(config.path, encoding='ISO-8859-1')
dictionary = {}
for i, row in enumerate(data_file.values):
    ######################################
    article_path, summary = row[4], row[2]
    # print(summary)
    article_path = article_path.replace(".xml", ".txt")
    article_path = "data/" + article_path
    # article_path = '/home/slab/PycharmProjects/pytorch_implementation/data'
    article_object = open(article_path, 'r', encoding='ISO-8859-1')
    article = article_object.readlines()
    #####################################
    article = [
        l.replace('  \n', '').replace('  #@NEW_LINE#@#', '').replace('title', '').replace('Abstract', '').replace(
            'Introduction', '') for l in article]
    article = [l for l in article if len(l) > 2]
    article = ''.join(article[1:5])
    # doc = nlp(article)
    # sent = [token.text for token in doc.sents][:config.n_input_sents]
    sent = nltk.sent_tokenize(article)[:config.n_input_sents]
    article = [val.lower() for sublist in sent for val in nltk.word_tokenize(sublist)]
    #####################################
    # doc = nlp(summary)
    # sent = [token.text for token in doc.sents][:config.n_output_sents]
    sent = nltk.sent_tokenize(summary)[:config.n_output_sents]
    summary = [val.lower() for sublist in sent for val in nltk.word_tokenize(sublist)]
    ######################################
    if (len(article) > config.max_enc_steps):
        article = article[:config.max_enc_steps]
    if (len(summary) > config.max_dec_steps):
        summary = summary[:config.max_dec_steps]

    pair = article + summary
    for w in pair:
        if w not in dictionary:
            dictionary[w] = len(dictionary)
    if (i % 100 == 0):
        print("Processed ", i, " records")

print("done creating dictionary of size= ", len(dictionary))
print("now writing to file.....")
#
vocab_file = open("data/vocab", "w", encoding='utf-8')
for k, v in dictionary.items():
    row = k + "#_#_#" + str(v) + "\n"
    vocab_file.write(row)
vocab_file.close()

print("Finished all activities!")
