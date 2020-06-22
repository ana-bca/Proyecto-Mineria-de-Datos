# PRIMER CODIGO ***************************************************************#
#================== Limpiesa y adecuacion de los datos ========================#
import pandas as pd
#=============================================================================#
# Los datos fueron descargados de https://www.yelp.com/dataset como parte de  #
# reto de yelp https://www.yelp.com/dataset/challenge/winner                  #
#=============================================================================#
#2) Lectura de la base de fotografias-----------------------------------------#
photos_json_path = 'photos.json'
df_photos = pd.read_json(photos_json_path, lines=True)
df_photos.shape
df_photos.head()
#3) Filtramos fotos de comida (label=food) -----------------------------------#
df_photos = df_photos[df_photos['label']=='food']
df_photos.shape
data_photos = df_photos.assign(length=df_photos.caption.map(len))
data_photos = data_photos.drop(['business_id'],axis=1)
# Trasformamos toda la informacion a formato csv para trabajar luego ----------#
csv_datos_finales = "datos_finales.csv"
data_photos.to_csv(csv_datos_finales , index=False,sep="\t")
#*****************************************************************************#
