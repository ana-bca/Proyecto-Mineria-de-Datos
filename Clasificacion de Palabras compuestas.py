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
#======================== Clasificador de bigramas ===========================#
vocabul = pd.read_csv('palabras_finales.csv', engine='python',sep="|")
# Vectores en los bigramas
data = pd.read_csv('palabrascompu2.csv',sep=",")
data_v = data[data.index<50]
data = data[data.index >= 50]


sentences = data['compuesta'].values
y = data['Clase'].values

vectorizer = CountVectorizer()
vectorizer.fit(sentences)
X = vectorizer.transform(sentences)

tokenizer = Tokenizer(num_words= table_n1.shape[0])
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X, padding='post', maxlen=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                    random_state=1000)


input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=1,
                    validation_split=0.1,
                    shuffle=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# Predicciones----------------------------------------------------------------#
y_pred = model.predict(X_test)
pd.DataFrame(data = {'pred':y_pred.reshape(100,),'test':y_test})


table_n2  = table_n2.assign(word = table_n2.index.map(lambda p: np.array(p)))
table_n2  = table_n2.assign(word = table_n2.index.map(lambda p: ' '.join(p)))
X_bigrams  = vectorizer.transform(table_n2.word.values)
X_bigrams  = tokenizer.texts_to_sequences(table_n2.word.values)

y_pred = model.predict(X_bigrams)
compuestas  = pd.DataFrame(data = {'word':table_n2.word.values
                                   'pred':y_pred.reshape(y_pred.shape[0],)})



