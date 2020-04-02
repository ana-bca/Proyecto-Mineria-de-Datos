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
business_json_path = 'dataset/yelp_academic_dataset_business.json'
df_business = pd.read_json(business_json_path, lines=True)
df_business.head(2)
photos_json_path = 'photos/photos.json'
df_photos = pd.read_json(photos_json_path, lines=True)
# Filtramos fotos de comida y seleccionamos los negocios de comida -------------#
df_photos = df_photos[df_photos['label']=='food']
df_business_food = df_business[df_business.business_id.isin(df_photos['business_id'])]
bssnes_ide =  ["business_id","stars","name","city","review_count","is_open"]
df_business_food = df_business_food.filter(items = bssnes_ide)
# Lectura de las opiniones correspondientes a los negocios de comidas ----------#
review_json_path = 'dataset/yelp_academic_dataset_review.json'
ifile = open(review_json_path) 
all_data = list()
for i, line in enumerate(ifile):
    if i%100==0:
        print(i)
    data  = json.loads(line)
    business_id = data['business_id']
    review_id = data['review_id']
    text = data['text']
    stars_review = data['stars']
    useful = data["useful"]
    funny = data["funny"]
    cool = data["cool"]
    all_data.append([review_id,business_id,text,stars_review,useful,funny,cool])

df_review = pd.DataFrame(all_data)
df_review = df_review.rename(columns={0: "business_id", 1: "review_id",2: "text",3: "stars_review",4:"userful",5:"funny",6:"cool"})

table_data = pd.merge(left=df_review, right=df_photos, left_on='business_id', right_on='business_id')
# Trasformamos toda la informacion a formato csv para trabajar luego -----------#
csv_datos_finales = "datos_finales.csv"
csv_review = "yelp_reviews.csv"
csv_business = "yelp_business.csv"
csv_photos = "yelp_photos.csv"
df_review.to_csv(csv_review, index=False)
df_business_food.to_csv(csv_business, index=False)
df_photos.to_csv(csv_photos, index=False)
table_data.to_csv(csv_datos_finales, index=False)


