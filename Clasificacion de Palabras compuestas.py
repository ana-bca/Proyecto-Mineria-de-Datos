import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#======================== Clasificador de bigramas ===========================#
vocabul = pd.read_csv('palabras_finales.csv', engine='python',sep="|")
# Vectores en los bigramas
data = pd.read_csv('palabrascompu2.csv',sep=",")

sentences = data['palabracompu'].values
sentences
y = data['Clase'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words= 991)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
from keras.preprocessing.sequence import pad_sequences
maxlen = 2
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=1,
                    validation_split=0.1,
                    shuffle=False,
                    validation_data=(X_test, y_test),
                    batch_size=15)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# Predicciones
y_pred = model.predict(X_test)
pd.DataFrame(data = {'pred':y_pred.reshape(100,),'test':y_test})

X_bigrams  = vectorizer.transform(sentences_test)
table_n2  = table_n2.assign(word = table_n2.index.map(lambda p: np.array(p)))
