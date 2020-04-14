#================== Limpiesa y adecuacion de los datos =========================#
import pandas as pd
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="darkgrid")
# Lectura de las bases negocios y comidas --------------------------------------#
business_json_path = 'yelp_academic_dataset_business.json'
df_business = pd.read_json(business_json_path, lines=True)
df_business.head(2)
photos_json_path = 'photos.json'
df_photos = pd.read_json(photos_json_path, lines=True)
# Filtramos fotos de comida y seleccionamos los negocios de comida -------------#
df_photos = df_photos[df_photos['label']=='food']
data_photo = df_photos.assign(length=df_photos.caption.map(len))
#- Seleccionamos negocios y seleccionamos numero de revisiones ----------------#
data_business = df_business[["business_id","stars"]][df_business.review_count > 50]
data_photo = pd.merge(data_photo,data_business,on='business_id',how='left')
data_photo = data_photo[data_photo.stars.notnull()]
data_photo = data_photo[data_photo.length > 3]
#Num_by_Bss = data_photo[["business_id"]].groupby("business_id").size()
#Num_by_Bss = Num_by_Bss[Num_by_Bss > 10]
#data_photo = data_photo[data_photo.business_id.isin(Num_by_Bss.index.values)]
#- Eliminar caracteres extraños en los textos ---------------------------------#
def removeT(s): return s.replace('|', '')
data_photo.caption = data_photo.caption.map(removeT)
# Trasformamos toda la informacion a formato csv para trabajar luego -----------#
csv_datos_finales = "datos_finales.csv"
data_photo.to_csv(csv_datos_finales , index=False,sep="|")
