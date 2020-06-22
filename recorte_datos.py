
#======================== Etiquetas mas frecuentes ===============================#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
import inflection
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
nlp = English()


#### lectura de datos
data = pd.read_csv('data_photo_F.csv', engine='python',sep="\t") ## Lectura de los datos
palabras = pd.read_csv('palabras_finales.csv', engine='python',sep="\t") ## Lectura de los datos
netique=40
etiquetas=palabras[:netique]
data.shape[0]

data_final=pd.DataFrame(columns=('photo_id', 'caption', 'label', 'length', 'stars','len_word'))
for i in range(data.shape[0]):
    cap= nltk.word_tokenize(data.caption[i]) 
    for j in range(len(cap)):
       for k in range(netique):
           if cap[j]==etiquetas.word[k]:
               data_final.loc[len(data_final)]=data.loc[i]

data_final.caption
data_final.shape
SQL_Query = pd.read_sql_query(
'''select photo_id,business_id,caption,label,length,stars,len_word'
from data_final
group by photo_id ''')
csv_datos_recortados = "data_final.csv"
data_final.to_csv(csv_datos_recortados, index=False,sep=",")

etiquetas.loc['cuenta']=0
data_final=pd.DataFrame(columns=('photo_id', 'caption', 'label', 'length', 'stars','len_word'))
for i in range(data.shape[0]):
    cap= nltk.word_tokenize(data.caption[i]) 
    for j in range(len(cap)):
       for k in range(netique):
           if cap[j]==etiquetas.word[k]:
               etiquetas.cuenta[k]=etiquetas.cuenta[k]+1
               if etiquetas.cuenta[k]<700:
                    data_final.loc[len(data_final)]=data.loc[i]

data_final.caption
etiquetas