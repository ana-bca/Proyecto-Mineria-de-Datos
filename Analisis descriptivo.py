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
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
# Lectura de los datos --------------------------------------------------------#
data = pd.read_csv('datos_finales.csv', engine='python',sep="|")
#==============================================================================#
regex = re.compile('[^A-Za-z0-9 ^. , ^: ^; ^? ^¿ ^¡ ^!]')
def texCC(x,rg): return rg.sub(' ',str(x))
# Analisis del corpus ---------------------------------------------------------#
data_F = data.assign(caption = data.caption.map(lambda p: texCC(p, regex)))
# Cantidad de palabras por comentario -----------------------------------------#
def LENW(x):return len(tknzr.tokenize(str(x)))
data_F = data_F.assign(len_word = data_F.caption.map(LENW))
pd.crosstab(data_F.len_word, columns='count').plot.bar()
plt.title("Cantidad de palabras por comentario")
plt.show()
# Analisis de n-gramas (frecuencias de una palabra) ---------------------------#
regex2 = re.compile('[^A-Za-z0-9]')
data_F2 = data.assign(caption = data.caption.map(lambda p: texCC(p, regex2)))
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
table_n1[table_n1.Count < 100].hist(bins = 100)

table_n1 = Table_NGRAM(data,1)

table_n1[table_n1.Count < 10][:30].plot.barh()

table_n2 = Table_NGRAM(data,2)
table_n2[:60].plot.barh()

table_n3 = Table_NGRAM(data,3)
table_n3[:60].plot.barh()

wordcloud = WordCloud(max_words=250,background_color="white").generate(data_F2.caption.sum())
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


