import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.utils import shuffle
#======================== Clasificador de bigramas ===========================#
vocabul = pd.read_csv('palabras_finales.csv', engine='python',sep="|")
# Vectores en los bigramas
#data = pd.read_csv('palabrascompu2.csv',sep=",")
data = pd.read_csv('prueba1.csv',sep=";")
data_A = shuffle(data)
data_v = data[data_A.index< int(data_A.shape[0]*0.10)]
data = data_A[data_A.index >= int(data_A.shape[0]*0.10)]
sentences = data['compuesta'].values
y = data['Clase'].values

tokenizer = Tokenizer(num_words= vocabul.shape[0])
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X, padding='post', maxlen=2)

X_exit = tokenizer.texts_to_sequences(data_v.compuesta.values)
X_exit = pad_sequences(X_exit, padding='post', maxlen=2)
y_exit = data_v['Clase'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=1000)


input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1,
                    shuffle=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_exit,y_exit, verbose=False)
print("Exit Accuracy:  {:.4f}".format(accuracy))

fig, axs = plt.subplots(1, 2)
axs[0].plot(history.history['loss'], label='train')
axs[0].plot(history.history['val_loss'], label='test')
axs[0].set_title('Loss')
axs[0].legend()
axs[1].plot(history.history['accuracy'], label='train')
axs[1].plot(history.history['val_accuracy'], label='test')
axs[1].set_title('Accuracy')
axs[1].legend()
# Predicciones----------------------------------------------------------------#
data_C = pd.read_csv('palabrascompu.csv',sep=",")
data_C = data_A.append(data_C)
data_C.drop_duplicates(subset ="compuesta", keep = False, inplace = True) 
data_C = data_C.sample(500)

y_C = data_C['Clase'].values
X_C = tokenizer.texts_to_sequences(data_C.compuesta.values)
X_C = pad_sequences(X_C, padding='post', maxlen=2)
y_pred = model.predict(X_C)
data_C['Clase'] = np.round(y_pred +0.05).astype(int)
data_C['Clase'].sum()
csv_datos_finales = "data_C.csv"
data_C.to_csv(csv_datos_finales, index=False,sep=",")