# CUARTO CODIGO **************************************************************#
#========================== PREDICCION =======================================#
import os
import keras
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style("white")

from sklearn.cluster import AgglomerativeClustering
from random import sample
# Codigo =====================================================================#
model = keras.models.load_model('my_model.h6') # Cargar modelo
## Graficas de Comportamiento de modelo (train-valid)-------------------------#
fig, axs = plt.subplots(1, 2)
axs[0].plot(history.history['loss'][1:], label='train')
axs[0].plot(history.history['val_loss'][1:], label='test')
axs[0].set_title('Loss')
axs[0].legend()
axs[1].plot(history.history['accuracy'][1:], label='train')
axs[1].plot(history.history['val_accuracy'][1:], label='test')
axs[1].set_title('Accuracy')
axs[1].legend()
## Clasificacion para identificar numero de etiquetas ------------------------#
### Creamos la tabla de predicciones ordenada en cada caso de menor a mayor
test_dir = 'Imagenes/test'
classes = train_generator.class_indices
decode_classes = {v: k for k, v in classes.items()}
decode_classes
filenames_full = []
for dirpath, dirnames, filenames in os.walk(test_dir):
    for filename in filenames:
        filenames_full.append(os.path.join(dirpath, filename))
filenames_full = filenames_full
rows = (len(filenames_full) - 1) // 4 + 1
Table_pred = []
names_photos = []
for index, filename_full in enumerate(filenames_full):
    plt.subplot(rows, 4, index + 1)
    names_photos.append(filenames_full)
    test_image = keras.preprocessing.image.load_img(filename_full, target_size=(200,250))
    test_input = keras.preprocessing.image.img_to_array(test_image) * (1. / 255)
    test_input = np.expand_dims(test_input, axis=0)
    prediction = model.predict(test_input)
    Table_pred.append(list(np.sort(prediction)[::-1]))
Table_pred = pd.DataFrame(np.vstack(Table_pred))
Table_pred.head()
### Realizamos un cluster en 3 grupos (1 etiqueta,2 etiquetas,3 etiquetas)
cluster = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage='ward')
cluster.fit_predict(Table_pred.iloc[:,17:])
unique_elements, counts_elements = np.unique(cluster.labels_, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
### Graficas boxplot para el comportamiento de las etiquetas en los grupos
Table_plot = Table_pred.iloc[:,17:].assign(clase = cluster.labels_)
fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10,8),sharex=True)
sns.boxplot(x = Table_plot.iloc[:,2],y = Table_plot.clase,orient="h",ax=ax1)
sns.boxplot(x = Table_plot.iloc[:,1],y = Table_plot.clase,orient="h",ax=ax2)
sns.boxplot(x = Table_plot.iloc[:,0],y = Table_plot.clase,orient="h",ax=ax3)
### Tabla resultado de los clusters 
Tab_Clus = pd.DataFrame({'name':filenames_full,'clase':cluster.labels_})
Tab_Clus.clase[Tab_Clus.clase == 2] = 3
Tab_Clus.clase[Tab_Clus.clase == 1] = 2
Tab_Clus.clase[Tab_Clus.clase == 0] = 1
Tab_Clus.head(10)
### Muestra de como predice en el conjunto de imagenes test
filenames_sample = sample(filenames_full,10)
rows = (len(filenames_sample) - 1) // 4 + 1
plt.figure(figsize=(15, 5 * rows))
for index, filename_samp in enumerate(filenames_sample):
    plt.subplot(rows, 4, index + 1)
    test_image = keras.preprocessing.image.load_img(filename_samp, 
                                                    target_size=(200,250))
    test_input = keras.preprocessing.image.img_to_array(test_image) * (1. / 255)
    test_input = np.expand_dims(test_input, axis=0)
    plt.imshow(test_image)
    plt.axis('off')
    prediction = model.predict(test_input)
    prediction = prediction.reshape(len(keys),)
    etiq = []
    n_etiq = int(Tab_Clus[Tab_Clus.name == filename_samp].clase)
    pred_aux = prediction.copy()
    type_name = []
    for jj in range(n_etiq):
      etiq.append(np.argmax(pred_aux))
      type_name.append(decode_classes[np.argmax(pred_aux)])
      pred_aux[np.argmax(pred_aux)] = 0
    print(prediction[etiq])
    plt.title(type_name)
#*****************************************************************************#

