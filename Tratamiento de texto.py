#======================== Tratamiento de Texto ===============================#
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
#================== Funciones Necesarias ======================================#
# Eliminar caracteres no alfabeticos
regex = re.compile('[^A-Za-z]')
def texClean1(x,rg): return rg.sub(' ',str(x))
# Eliminar palabras vacias
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
    return ' '.join(filtered_sentence)
# Poner todo en Singular
def SingTex(x):
    sentence = str(x).split()
    text = [inflection.singularize(wrd) for wrd in sentence]
    result = ' '.join(text)
    return result
# Eliminar palabras segun diccionario
Dic1 = pd.read_csv('palabras1.csv', engine='python',sep=",")
Dic1 = np.array(Dic1[Dic1.Ind.isnull()].word)

Dic2 = pd.read_csv('palabras2.csv', engine='python',sep="\t")
Dic2 = np.array(Dic2[Dic2.Ind.isnull()].word)

Dic3 = pd.read_csv('palabras3.csv', engine='python',sep="\t")
Dic3 = np.array(Dic3)
Dic3 = Dic3.reshape(Dic3.shape[0],)

DicF = np.append(np.append(Dic1,Dic2),Dic3)

def texClean3(x,DIC):
    x = str(x)
    querywords = x.split()
    resultwords  = [word for word in querywords if word.lower() not in DIC]
    result = ' '.join(resultwords)
    return result
def texClean4(x):
    x = str(x)
    resultwords = list(dict.fromkeys(x.split()))
    result = ' '.join(resultwords)
    return result
# longitud de un texto
def LENW(x):return len(tknzr.tokenize(str(x)))
## Tratamiento completo de el texto
def TratTex(x):
    text = str(x).lower()
    text = texClean1(text,regex)
    text = texClean2(text)
    text = SingTex(text)
    text = texClean3(text,DicF)
    text = texClean4(text)
    return text
# Lectura de los datos --------------------------------------------------------#
data = pd.read_csv('datos_finales.csv', engine='python',sep="|")
# Tratamiento de el texto -----------------------------------------------------#
data_F = data.assign(caption = data.caption.map(lambda p: TratTex(p)))
# Eliminacion de Textos sin contenido -----------------------------------------#
data_F = data_F.assign(len_word = data_F.caption.map(LENW))
data_F = data_F[data_F.len_word != 0]
pd.crosstab(data_F.len_word, columns='count').plot.bar()
plt.title("Cantidad de palabras por comentario")
plt.show()
#Creaccion de diccionario de palabras------------------------------------------#
def NGRAM(x,n):
    token=nltk.word_tokenize(str(x))
    bigrams=ngrams(token,n)
    return list(bigrams)
def Table_NGRAM(data,n):
    n_grams = data.caption.map(lambda p: NGRAM(p,n))
    n_grams = sum(n_grams,[])
    table_n = pd.DataFrame(FreqDist(n_grams), index =[0]).T
    table_n.columns = ['Count']
    table_n = table_n.sort_values('Count', ascending = False)
    return table_n
# n-gramss n = 1 ..............................................................#
table_n1 = Table_NGRAM(data_F,1)
table_n1[:40].plot.barh()
table_n1  = table_n1.assign(word = table_n1.index.map(lambda p: np.array(p)))
table_n1.word = [val for sublist in table_n1.word for val in sublist]
csv_name = "palabras_finales.csv"
table_n1.to_csv(csv_name, index=False,sep="|")
# n-gramss n = 1 ..............................................................#
table_n2 = Table_NGRAM(data_F,2)
table_n2[:40].plot.barh()
palabra1 = np.array(table_n2.index.map(lambda p: p[0]))
palabra2 = np.array(table_n2.index.map(lambda p: p[1]))
compuesta = np.array(table_n2.index.map(lambda p: ' '.join(np.array(p))))
palabrascompu = pd.DataFrame({'palabra1':palabra1,'palabra2':palabra2,
                              'compuesta':compuesta,
                              'Clase':np.zeros(palabra1.shape[0],dtype= int)})
csv_datos_finales = "palabrascompu.csv"
palabrascompu.to_csv(csv_datos_finales, index=False,sep=",")
# n-gramss n = 1 ..............................................................#
table_n3 = Table_NGRAM(data_F,3)
table_n3[:30].plot.barh()
#==============================================================================#
table_n1  = table_n1.assign(word = table_n1.index.map(lambda p: np.array(p)))
table_n1.word = [val for sublist in table_n1.word for val in sublist]
DIC3 = np.array(table_n1.word[table_n1.Count < 4])
csv_datos_finales = "palabras3.csv"
pd.DataFrame(DIC3).to_csv(csv_datos_finales, index=False,sep="|")
# Exportar listas de palabras para evaluar diccionarios -----------------------#
csv_datos_finales = "palabras2.csv"
csv_datos_finales = "palabras1.csv"
table_n1[:7000][:].to_csv(csv_datos_finales , index=False,sep="|")
table_n1[6999:][:].to_csv(csv_datos_finales , index=False,sep="|")

