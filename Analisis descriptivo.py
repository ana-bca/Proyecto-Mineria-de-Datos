#=================== Analisis descriptivo de la base ==========================#
import pandas as pd
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Lectura de los datos --------------------------------------------------------#
data = pd.read_csv('datos_finales.csv', engine='python',sep="|")
#==============================================================================#
#=========================== ANALISIS DE PHOTOS ===============================#
#-Longitud de los comnetarios en la photo--------------------------------------#
data.length.hist(bins=50)
#-Numero de photos por business -----------------------------------------------#
Num_by_Bss = data[["business_id"]].groupby("business_id").size()
pd.crosstab(Num_by_Bss, columns='count').plot.bar()
#- Analisis de la longitud de texto por estrellas -----------------------------#
pd.crosstab(data.stars, columns='count').plot.bar()
sns.boxplot(x='stars', y='length', data=data)
data[["stars","length"]].groupby("stars").describe().reset_index()
#- Calculo de sentimientos y filtro por opiniones objetivas -------------------#
from textblob import TextBlob
import re
regex = re.compile('[^A-Za-z0-9 ^. , ^: ^; ^? ^¿ ^¡ ^!]')
def texCC(x): regex.sub(' ',str(x))
data_photo.caption = data.caption.map(texCC)
# Analisis de sentimientos
def Tf(x):
    val = TextBlob(str(x)).sentiment.subjectivity
    return val
def TfP(x):
    val = TextBlob(str(x)).sentiment.polarity
    return val
data = data.assign(subjectivity =data.caption.map(Tf)) # Tomamos comentarios subjetivos
data[data.subjectivity != 0].subjectivity.hist(bins=20)
sns.boxplot(x='stars', y='subjectivity', data=data[data.subjectivity != 0])

data = data.assign(polarity =data.caption.map(TfP)) # Polaridad de las opiniones
data[data.subjectivity > 0.3].polarity.hist(bins=20)
sns.boxplot(x='stars', y='polarity', data=data[data.subjectivity > 0.3])
