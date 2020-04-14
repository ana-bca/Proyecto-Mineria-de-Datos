#=================== Analisis descriptivo de la base ==========================#
import pandas as pd
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
### hola andrey, y hola mundo con coronavirus
# Lectura de los datos --------------------------------------------------------#
data = pd.read_csv("yelp_photos.csv")
data.head(2)
## Numero de reseñas por negocios 
Num_by_Bss = data[["review_id","business_id"]].groupby("business_id").size()
pd.crosstab(Num_by_Bss, columns='count').plot.bar()
plt.xlabel('Numero de opiniones')
plt.title("Opiniones por negocios")
plt.legend.remove()
## Numero de reseñas por fotografia
Num_by_Photo = data[["review_id","photo_id"]].groupby("photo_id").size()
## Comportamiento de las estrellas 
### Comportamiento general de las estrellas
pd.crosstab(data.stars_review, columns='count').plot.bar()
plt.legend.remove()
### Comportamiento de longitud de texto por estrellas
data_len = data.assign(length=data.text.map(len))
graph = sns.FacetGrid(data=data_len,col='stars_review')
graph.map(plt.hist,'length',bins=50,color='blue')
#------------------------------------------------------------------------------#
# Parece que, en general, la distribución de la longitud del texto es similar
# en las cinco clasificaciones. Sin embargo, el número de revisiones de texto
# parece estar sesgado mucho más hacia las calificaciones de 4 y 5 estrellas. 
# Esto puede causar algunos problemas más adelante en el proceso
#------------------------------------------------------------------------------#
sns.boxplot(x='stars_review', y='length', data=data_len)
#==============================================================================#
#=========================== ANALISIS DE PHOTOS ===============================#
#-Longitud de los comnetarios en la photo--------------------------------------#
data_photo = df_photos.assign(length=df_photos.caption.map(len))
data_photo.length[data_photo.length != 0].hist(bins=50)
#-Numero de photos por business -----------------------------------------------#
Num_by_Bss = data_photo[["business_id"]].groupby("business_id").size()
pd.crosstab(Num_by_Bss, columns='count').plot.bar()
#- Seleccionamos negocios y seleccionamos numero de revisiones ----------------#
NRw = pd.crosstab(df_business.review_count[df_business.review_count > 50],columns='count')
NRw[:100].plot.bar()
data_business = df_business[["business_id","stars"]][df_business.review_count > 50]
pd.crosstab(data_business.stars, columns='count').plot.bar()
#- Cruzamos data de fotos con la data de estrellas ----------------------------#
data_photo = pd.merge(data_photo,data_business,on='business_id',how='left')
data_photo = data_photo[data_photo.stars.notnull()]
#- Analisis de la longitud de texto por estrellas
pd.crosstab(data_photo.stars, columns='count').plot.bar()
data_photo.length[data_photo.length != 0][data_photo.stars == 3.0].hist(bins=50)
data_photo[["stars","length"]][data_photo.length != 0].groupby("stars").describe().reset_index()

data_photo.caption[data_photo.length != 0][data_photo.stars == 1.0][:1]
ff = data_photo[data_photo.length > 3]
pd.crosstab(ff.stars, columns='count').plot.bar()
