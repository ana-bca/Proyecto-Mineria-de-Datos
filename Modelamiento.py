# TERCER CODIGO **************************************************************#
#======================== MODELAMIENTO =======================================#
import numpy as np
#!pip install scikit-image
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import nltk
nltk.download('punkt')
from nltk.util import ngrams
from spacy.lang.en import English
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
nlp = English()

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
# Funciones ==================================================================#
def TUPT(x):
    x_N1 = np.array(NGRAM(x,1))
    x_N1 = x_N1.reshape(x_N1.shape[0],)
    x_T = list(set(x_N1.astype(str)))
    return x_T
def NGRAM(x,n):
    token=nltk.word_tokenize(str(x))
    bigrams=ngrams(token,n)
    return list(bigrams)
# Lectura de tablas ==========================================================#
data_train = pd.read_csv('Imagenes/data_train.csv', engine='python',sep="\t")
data_test = pd.read_csv('Imagenes/data_test.csv', engine='python',sep="\t")
data_valid = pd.read_csv('Imagenes/data_valid.csv', engine='python',sep="\t")
Ngrams_1 = pd.read_csv('palabras_finales.csv', engine='python',sep="\t")
# Codigo =====================================================================#
## Creacion de directorios con test-train-valid-------------------------------#
data_photo_F = pd.read_csv('data_photo_F.csv', engine='python',sep="\t")
data_photo_F = shuffle(data_photo_F)
data_photo_F = data_photo_F[['photo_id','caption',]]
data_photo_F.photo_id = data_photo_F.photo_id  + '.jpg'
### Separacion de las imagenes en entrenamiento, test y validacion
data_train, data_test = train_test_split(data_photo_F,test_size=0.2,random_state=1000)
data_train, data_valid = train_test_split(data_train,test_size=20/80,random_state=1000)
csv_train = "Imagenes/data_train.csv"
csv_test = "Imagenes/data_test.csv"
csv_valid = "Imagenes/data_valid.csv"
data_train.to_csv(csv_train , index=False,sep="\t")
data_test.to_csv(csv_test , index=False,sep="\t")
data_valid.to_csv(csv_valid , index=False,sep="\t")
#### Creamos los directorios para las imagenes
for i in tqdm.tqdm(data_test.photo_id.values):
    Imagen = io.imread('/home/andrey/Descargas/yelp_photos (1)/photos/' + i)
    image_resized = resize(Imagen, (200,250),anti_aliasing=True)
    local = 'Imagenes/test/' + i
    io.imsave(local,image_resized)
for i in tqdm.tqdm(data_train.photo_id.values):
    Imagen = io.imread('/home/andrey/Descargas/yelp_photos (1)/photos' + i)
    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
    local = 'Imagenes/train/' + i
    io.imsave(local,image_resized)
for i in tqdm.tqdm(data_valid.photo_id.values):
    Imagen = io.imread('/home/andrey/Descargas/yelp_photos (1)/photos' + i)
    image_resized = resize(Imagen, (400,500),anti_aliasing=True)
    local = 'Imagenes/valid/' + i
    io.imsave(local,image_resized)
#-----------------------------------------------------------------------------#
# Modelo con keras -----------------------------------------------------------#
## Generamos las clases en los dataframe
data_train = data_train.assign(tags = np.array(data_train.caption.map(lambda p: TUPT(p))))
data_test = data_test.assign(tags = np.array(data_test.caption.map(lambda p: TUPT(p))))
data_valid = data_valid.assign(tags = np.array(data_valid.caption.map(lambda p: TUPT(p))))    
## Cree un ImageDataGenerator con flow_from_dataframe-------------------------#
## Vector con las clases en los datos
Ngrams_1 = Ngrams_1.word.values
keys = list(set(Ngrams_1.astype(str)))
len(keys) # Numero de clases en los datos

datagen_train = ImageDataGenerator(
    rescale = 1./255,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range = 5,
    horizontal_flip=True)

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen_train.flow_from_dataframe(
    dataframe = data_train[["photo_id","tags"]],
    directory = "Proyecto-Mineria-de-Datos/Imagenes/train",
    x_col = "photo_id",
    y_col = "tags",
    batch_size = 32,
    class_mode = "categorical",
    shuffle=True,
    color_mode = "rgb",
    classes = keys,
    target_size = (200,250))

valid_generator = datagen.flow_from_dataframe(
    dataframe = data_valid[["photo_id","tags"]],
    directory = "Proyecto-Mineria-de-Datos/Imagenes/valid",
    x_col = "photo_id",
    y_col = "tags",
    batch_size = 32,
    class_mode = "categorical",
    shuffle=True,
    color_mode = "rgb",
    classes = keys,
    target_size = (200,250))

test_generator = datagen.flow_from_dataframe(
    dataframe = data_test[["photo_id","tags"]],
    directory = "Proyecto-Mineria-de-Datos/Imagenes/test",
    x_col = "photo_id",
    y_col = "tags",
    batch_size = 32,
    class_mode = None,
    shuffle=True,
    color_mode = "rgb",
    target_size = (200,250))
# Construccion de el modelo --------------------------------------------------#
EPOCHS = 30
INIT_LR = 1e-3
BS = 32

model = Sequential()
inputShape = (200,250,3)
chanDim = -1
if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
# CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# softmax classifier
model.add(Dense(len(keys)))
model.add(Activation("sigmoid"))
# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# Entrenamiento de el modelo (10 min por epoc) -------------------------------#
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30
)
# Guardamos el modelo final
model.save('my_model.h6')
#*****************************************************************************#
    
