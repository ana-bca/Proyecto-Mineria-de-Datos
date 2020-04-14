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