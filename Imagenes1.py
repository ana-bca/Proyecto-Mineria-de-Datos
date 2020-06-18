import numpy as np
import matplotlib.pyplot as plt
#!pip install scikit-image
from keras.preprocessing.image import img_to_array
from skimage import data
from skimage.color import rgb2gray
from skimage import io
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
import nltk
from nltk.util import ngrams
from spacy.lang.en import English
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
nlp = English()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
# Funciones ==================================================================#
# Lectura de tablas ==========================================================#
data_F = pd.read_csv('data_F.csv', engine='python',sep=",")
data_train = pd.read_csv('Imagenes/data_train.csv', engine='python',sep=",")
data_test = pd.read_csv('Imagenes/data_test.csv', engine='python',sep=",")
data_valid = pd.read_csv('Imagenes/data_valid.csv', engine='python',sep=",")

Ngrams_1 = pd.read_csv('palabras_finales.csv', engine='python',sep="|")
Ngrams_2 = pd.read_csv('prueba1.csv',sep=",")
# Funciones ==================================================================#
N2 = Ngrams_2.compuesta.values
def TUPT(x):
    x_N1 = np.array(NGRAM(x,1))
    x_N1 = x_N1.reshape(x_N1.shape[0],)
    #x_N2 = pd.DataFrame(np.array(NGRAM(data_F.caption[0],2)))
    #x_N2 = x_N2.agg(' '.join, axis=1).values
    #x_N2 = np.array([x for x in x_N2 if x in N2])
    #x_T = list(np.append(x_N1.astype(str), x_N2.astype(str)))
    x_T = list(set(x_N1.astype(str)))
    return x_T
def NGRAM(x,n):
    token=nltk.word_tokenize(str(x))
    bigrams=ngrams(token,n)
    return list(bigrams)
# Codigo =====================================================================#
#data_F = shuffle(data_F)
#data_F = data_F[['photo_id','caption',]]
#data_F.photo_id = data_F.photo_id  + '.jpg'
# Separacion de las imagenes en entrenamiento, test y validacion
#data_train, data_test = train_test_split(data_F,test_size=0.2,random_state=1000)
#data_train, data_valid = train_test_split(data_train,test_size=20/80,random_state=1000)
#csv_train = "Imagenes/data_train.csv"
#csv_test = "Imagenes/data_test.csv"
#csv_valid = "Imagenes/data_valid.csv"
#data_train.to_csv(csv_train , index=False,sep=",")
#data_test.to_csv(csv_test , index=False,sep=",")
#data_valid.to_csv(csv_valid , index=False,sep=",")

# Creamos los directorios para las imagenes
for i in tqdm.tqdm(data_test.photo_id.values):
    Imagen = io.imread('photos_F/' + i)
    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
    local = 'Imagenes/test/' + i
    io.imsave(local,image_resized)
for i in tqdm.tqdm(data_train.photo_id.values):
    Imagen = io.imread('photos_F/' + i)
    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
    local = 'Imagenes/train/' + i
    io.imsave(local,image_resized)
for i in tqdm.tqdm(data_valid.photo_id.values):
    Imagen = io.imread('photos_F/' + i)
    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
    local = 'Imagenes/valid/' + i
    io.imsave(local,image_resized)
# Generamos las clases en los dataframe
data_train = data_train.assign(tags = np.array(data_train.caption.map(lambda p: TUPT(p))))
data_test = data_test.assign(tags = np.array(data_test.caption.map(lambda p: TUPT(p))))
data_valid = data_valid.assign(tags = np.array(data_valid.caption.map(lambda p: TUPT(p))))
# Cree un ImageDataGenerator con flow_from_dataframe
## Clases en los datos
Ngrams_1 = Ngrams_1.word.values
#Ngrams_2 = Ngrams_2.compuesta[Ngrams_2.Clase == 1].values
keys =   np.append(Ngrams_1, Ngrams_2) # Estas son las clases definitivas
keys = list(set(Ngrams_1.astype(str)))

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe = data_train[["photo_id","tags"]].sample(1000),
    directory = "/home/andrey/Documentos/GitHub/Proyecto-Mineria-de-Datos/Imagenes/train",
    x_col = "photo_id",
    y_col = "tags",
    batch_size = 32,
    class_mode = "categorical",
    color_mode = "rgb",
    classes = keys,
    target_size = (400,500))

valid_generator = datagen.flow_from_dataframe(
    dataframe = data_valid[["photo_id","tags"]].sample(500),
    directory = "/home/andrey/Documentos/GitHub/Proyecto-Mineria-de-Datos/Imagenes/valid",
    x_col = "photo_id",
    y_col = "tags",
    batch_size = 32,
    class_mode = "categorical",
    color_mode = "rgb",
    classes = keys,
    target_size = (400,500))

test_generator = datagen.flow_from_dataframe(
    dataframe = data_test[["photo_id","tags"]].sample(500),
    directory = "/home/andrey/Documentos/GitHub/Proyecto-Mineria-de-Datos/Imagenes/test",
    x_col = "photo_id",
    y_col = "tags",
    batch_size = 32,
    class_mode = None,
    color_mode = "rgb",
    target_size = (400,500))

# Construye el modelo:
model = Sequential()
model.add(Conv2D(32,(3,3),padding = 'same',input_shape = (400,500,3))) 
model.add(Activation ('relu'))
model.add(Conv2D(32,(3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size = (2,2))) 
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),padding = 'same')) 
model.add(Activation('relu')) 
model.add(Conv2D(64,(3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size = (2,2))) 
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout (0.5))
model.add(Dense(983,activation='sigmoid'))
model.compile(optimizers.rmsprop(lr = 0.0001,decay = 1e-6),
              loss = "binary_crossentropy", metrics = ["accuracy"])
# Fitting the Model
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=4
)
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)