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

import stanfordnlp
stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline(lang='en')
nlp = stanfordnlp.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

cadena_cats = data_F.caption[160]
doc = nlp(cadena_cats)
doc.sentences[0].print_dependencies()


cadena_cats = str(data.caption[160])
doc_review = nlp(cadena_cats)
filtered_wr =[]
for sent in doc_review.sentences: # recorremos cada oración
    for dep in sent.dependencies: # recorremos cada palabra
        if dep[1] == 'amod': # el subíndice 1 indica que se trata de la clase funcional
           filtered_wr.append([dep[0].text, dep[2].text])
filtered_wr         
            
cadena_cats = “Cats fue una película realmente terrible. Esa película fue un desastre natural.”
>>> doc = nlp(cadena_cats)
            
doc_review.sentences[0].print_dependencies()   
            
review = "Sin embargo, no las tiene todas consigo Zombie y es, precisamente, cuando tiene que lograr mantener la tensión tras el impetuoso retorno cuando patina. El director pierde el pulso vibrante para dejarse llevar por una trama más apagada, sin chispa y del todo previsible. El segundo tercio del film se desploma. Ese montaje paralelo donde vemos, de un lado a Strode intentando reponerse y a Myers resucitando de sus cenizas y deambulando cual sombra por el campo en un dilatado regreso a Haddonfield, se desinfla. Previsible, aburrido, sin pulso. Y dejando entrever las costuras de un guión en el que su creador no logra sostener el ritmo. Las pesadillas de Strode y los asesinatos de Myers se vuelven aburridos. Y no solo por repetición insulsa sino también porque Zombie se desata con escenas más creativas, pero carentes de originalidad. Donde ese doble juego entre realidad y ensoñación se vuelve insistente, pierde sutileza para convertirse en una evidente falta de recursos e imaginación. Aquí debería haber dejado su impronta y solo consigue rozar lo convencional, incluso lo ridículo. Basta comprobar las apariciones del Dr. Loomis (Malcolm McDowell) que rozan lo cómico o la insistencia con la que nos quiere recordar ese sueño de Myers con angelical madre y el caballo blanco, que acaban tomando un protagonismo excesivo y perdiendo toda la sugerencia que debería acompañar para convertirse en una incómoda y repetida presencia que acaba entorpeciendo el avance de la historia en varias escenas. Lamentablemente el desagisado hace llegar al tramo final sin muchas esperanzas de sorpresa. Y menos aún de tensión. El largo regreso de Myers y el desenlace ya no recuperan la brillantez inicial. Y muy al contrario demuestra que quizás Zombie dispuso de una gran oportunidad para reivindicarse pero que no supo aprovechar."
