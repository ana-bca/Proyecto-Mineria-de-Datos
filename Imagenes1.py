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

id_photo = data_F.photo_id.values + '.jpg'
for i in tqdm.tqdm(range(id_photo.shape[0])):
    Imagen = io.imread('/home/andrey/Documentos/GitHub/yelp_photos/photos/' + id_photo[i])
    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
    io.imsave('/home/andrey/Documentos/GitHub/yelp_photos/photos_F/' + id_photo[i],image_resized)