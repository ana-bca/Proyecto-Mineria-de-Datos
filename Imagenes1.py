import numpy as np
import matplotlib.pyplot as plt
#!pip install scikit-image
from skimage import data
from skimage.color import rgb2gray
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import tqdm

data_F = pd.read_csv('data_F.csv', engine='python',sep=",")
#id_photo = data_F.photo_id.values + '.jpg'
#for i in tqdm.tqdm(range(id_photo.shape[0])):
#    Imagen = io.imread('/home/andrey/Documentos/GitHub/yelp_photos/photos/' + id_photo[i])
#    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
#    io.imsave('/home/andrey/Documentos/GitHub/yelp_photos/photos_F/' + id_photo[i],image_resized)
data_F = data_F.sample(5000)
# Cargamos las etiquetas definitivas    
Ngrams_1 = pd.read_csv('palabras_finales.csv', engine='python',sep="|")
Ngrams_1 = Ngrams_1.word.values
Ngrams_2 = pd.read_csv('prueba1.csv',sep=",")
Ngrams_2 = Ngrams_2.compuesta[Ngrams_2.Clase == 1].values
keys =   np.append(Ngrams_1, Ngrams_2) # Estas son las clases definitivas
keys = shuffle(keys).tolist()
cv1 = CountVectorizer(vocabulary = keys, ngram_range=(1, 2))
Y = cv1.transform(data_F.caption)
#cv1.inverse_transform(Y.toarray()[0])
id_photo = data_F.photo_id.values + '.jpg'
Imagen = io.imread('/home/andrey/Documentos/GitHub/yelp_photos/photos_F/' + id_photo[1])
io.imshow(Imagen)
io.show()