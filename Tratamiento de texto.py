#=================== Analisis descriptivo de la base ==========================#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from textblob import TextBlob
import re
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
from nltk.probability import FreqDist
# Lectura de los datos --------------------------------------------------------#
data = pd.read_csv('datos_finales.csv', engine='python',sep="|")
data_F = data
# Ponemos los datos en minusculas
data_F.caption = data_F.caption.str.lower()
# Eliminamos caracteres no alfabeticos
regex = re.compile('[^A-Za-z]')
def texClean1(x,rg): return rg.sub(' ',str(x))
data_F = data_F.assign(caption = data_F.caption.map(lambda p: texClean1(p, regex)))
# Eliminar palabras vacias
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
nlp = English()
def texClean2(x):
    my_doc = nlp(str(x))
    token_list = []
    for token in my_doc:
        token_list.append(token.text)
    filtered_sentence =[]
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)
data_F = data_F.assign(caption = data_F.caption.map(lambda p: texClean2(p)))
#    return filtered_sentence
#Creaccion de diccionario de palabras============================================#
Dic_New =[]
def NGRAM(x,n):
    token=nltk.word_tokenize(str(x))
    bigrams=ngrams(token,n)
    return list(bigrams)
def Table_NGRAM(data,n):
    n_grams = data.caption.map(lambda p: NGRAM(p,1))
    n_grams = sum(n_grams,[])
    table_n = pd.DataFrame(FreqDist(n_grams), index =[0]).T
    table_n.columns = ['Count']
    table_n = table_n.sort_values('Count', ascending = False)
    return table_n
table_n1 = Table_NGRAM(data_F,1)
table_n1[table_n1.Count < 10][:30].plot.barh()
