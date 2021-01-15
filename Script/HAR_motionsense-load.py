#!/usr/bin/env python
# coding: utf-8

# ## Working code

# In[1]:


import numpy as np
from numpy import fft
import pandas as pd
import seaborn as sns
import pickle
import glob
import matplotlib.pyplot as plt
from scipy import stats, fftpack

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical, vis_utils, normalize
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Input, RepeatVector, TimeDistributed


# In[5]:


root = 'datasets/motion-sense-master/data/'
window_size = 250
time_step  = 2


# In[3]:


# with open(root+"motionsense_dct_data_6_250.pkl", "rb") as f:
#     dct_test_data = np.asarray(pickle.loads(f.read()))
# with open(root+"motionsense_fft_data_6_250.pkl", "rb") as f:
#     fft_test_data = np.asarray(pickle.loads(f.read()))
# with open(root+"motionsense_features_6_250.pkl", "rb") as f:
#     test_features = np.asarray(pickle.loads(f.read()))
# with open(root+"motionsense_label_6_250.pkl", "rb") as f:
#     label = np.asarray(pickle.loads(f.read()))


# In[6]:


# Function to perform classification using SVM

def classification_Using_SVM(features, label):
    x_train,x_test,y_train,y_test=train_test_split(features, 
                                               label,
                                               test_size=0.20,
                                               random_state=0)

    svm_classifier = svm.SVC(kernel = 'rbf')
    svm_classifier.fit(x_train, y_train)
    return svm_classifier.predict(x_test), y_test

#Function to plot the confusion matrix in heat map format
def confusion_Matrix(y_test, predict, activities, title):
    
    cf_matrix = metrics.confusion_matrix(y_test, predict, labels=activities)
    plt.figure(figsize=(15,10))
    cm = sns.heatmap(cf_matrix, annot=True, cmap='Reds', fmt='g', 
                xticklabels=activities, yticklabels=activities)
    plt.title("Confusion matrix for"+title, fontsize=25)
    plt.xlabel("Predicted activities", fontsize=20)
    plt.ylabel("Actual activities", fontsize=20)
    plt.xticks(fontsize= 15, rotation=90)
    plt.yticks(fontsize= 15, rotation=0)
    plt.savefig('Motionsense_confusion_matrix_'+title+"_"+str(time_step)+"_"+str(window_size)+'.png',
                bbox_inches = 'tight')
    plt.show()
    


# In[ ]:


# test_features = np.hstack((mean_test_data, std_test_data, var_test_data,
#                           cor_test_data, abdev_test_data, maxpeak_test_data, 
#                           minpeak_test_data, fft_test_data ))


# label = np.asarray(label)

# predict, y_test = classification_Using_SVM(np.nan_to_num(test_features), label)


# In[7]:


test_activities = ['Walking', 'Jogging', 'Sitting',
                   'Standing', 'Upstairs', 'Downstairs']


# In[10]:


# predict, y_test = classification_Using_SVM(np.nan_to_num(test_features), label)
# print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
# print(metrics.classification_report(y_test, predict, labels=test_activities))


# In[ ]:


with open("motionsense_data_"+str(time_step)+"_"+str(window_size)+".pkl", "rb") as f:
    test_data = np.asarray(pickle.loads(f.read()))

with open("motionsense_label_"+str(time_step)+"_"+str(window_size)+".pkl", "rb") as f:
    label = np.asarray(pickle.loads(f.read()))

print("Shape of the data after sliding window")
print(test_data.shape)


# In[ ]:


def autoencoder(total_data, activation_fn, num_features, epoch, batch_n):
    input_layer = Input(shape=(total_data.shape[1],total_data.shape[2], ))
    encoder = LSTM(num_features, activation=activation_fn, kernel_initializer="he_uniform")(input_layer)
    #encoder = LSTM(180, activation='sigmoid')(encoder)
    decoder = RepeatVector(total_data.shape[1])(encoder)
    #decoder = LSTM(96, return_sequences=True, 
    #               activation='sigmoid')(decoder)
    decoder = LSTM(total_data.shape[2], return_sequences=True, 
                   activation=activation_fn, kernel_initializer="he_uniform")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
#     output = TimeDistributed(Dense(total_data.shape[2]))(decoder) 
    
#     autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.summary()
    encoderModel = Model(input_layer, encoder)
    autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
    autoencoder.fit(total_data, total_data, epochs = epoch, batch_size = batch_n, validation_split=0.2, verbose=1)
    encoded_data = encoderModel.predict(total_data)
    autoencoder.save("motionsense_my_autoencoder.h5")
    encoderModel.save("motionsense_my_encoder.h5")
    return encoded_data


def autoencoder2l(total_data, activation_fn, num_features, epoch, batch_n):
    input_layer = Input(shape=(total_data.shape[1],total_data.shape[2], ))
    encoder = LSTM(6, return_sequences= True, activation=activation_fn)(input_layer)
    encoder = LSTM(30, activation=activation_fn)(encoder)
    decoder = RepeatVector(total_data.shape[1])(encoder)
    decoder = LSTM(30, return_sequences=True, activation=activation_fn)(decoder)
    decoder = LSTM(6, return_sequences=True, 
                   activation=activation_fn)(decoder)
    output = TimeDistributed(Dense(total_data.shape[2]))(decoder) 
    print(total_data.shape[2])
    autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.summary()
    encoderModel = Model(input_layer, encoder)
    autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
    autoencoder.fit(total_data, total_data, epochs = epoch, batch_size = batch_n, validation_split=0.2, verbose=1)
    encoded_data = encoderModel.predict(total_data)
    return encoded_data


# In[ ]:


# Function to build the LSTM model

def classification_Using_LSTM(features, label):
    X_local = []
    Y_seq = []
    sequence_length = 3
    step=1
    # Converting the 2-D data to 3-D for LSTM
    for start in range(0, len(features) - sequence_length, step):
        end = start + sequence_length
        X_local.append(features[start:end])
        Y_seq.append(label[end-1])
        
    # Converting class labels to numbers for processing
    Y_local = pd.factorize(Y_seq)[0].tolist()
    X_sequence = np.array(X_local)
    Y = np.array(Y_local)
    
    #Initiating the model
    model = Sequential()
    #Building the hidden layer with LSTM
    model.add(LSTM(300, input_shape = (X_sequence.shape[1], X_sequence.shape[2])))
    #Using Dropout to avoid overfitting
    model.add(Dropout(0.5))
    #Building the output layer
    model.add(Dense(6, activation="sigmoid"))
    #Compiling the model
    model.compile(loss="sparse_categorical_crossentropy"
                  , metrics=['accuracy']
                  , optimizer="adam")

    print(model.summary())
    #Spliting the data for training, validation and testing
    X_sequence = np.array(X_local)
    Y = np.array(Y_local)
    training_size = int(len(X_sequence) * 0.8)
#     validation_size = int((len(X_sequence)-training_size)/2)
    X_train, y_train = X_sequence[:training_size], Y_local[:training_size]
#     X_valid, y_valid = X_sequence[training_size:training_size + validation_size], Y_local[training_size:training_size + validation_size]
    X_test, y_test = X_sequence[training_size:], Y_local[training_size:]
    #Fit the model
    model.fit(X_train, y_train, batch_size=4, epochs=60, validation_split=0.2, verbose=1)
    # Evaluate
    model.evaluate(X_test, y_test)
    # Predict
    y_test_prob = model.predict(X_test, verbose=1)
    integers = [np.argmax(np.array(vector)) for vector in y_test_prob]
    predict_dataframe = pd.DataFrame(integers, columns=['activity'])
    y_test_dataframe = pd.DataFrame(y_test, columns=['activity'])
    # Mapping the class labels back to string
    predict_dataframe['activity'] = predict_dataframe['activity'].map({0:'Downstairs',
                                                                       1:'Upstairs',
                                                                       2:'Walking',
                                                                       3:'Jogging',
                                                                       4:'Standing',
                                                                       5:'Sitting'})
    y_test_dataframe['activity'] = y_test_dataframe['activity'].map({0:'Downstairs',
                                                                     1:'Upstairs',
                                                                     2:'Walking',
                                                                     3:'Jogging',
                                                                     4:'Standing',
                                                                     5:'Sitting'})
    return y_test_dataframe, predict_dataframe


# In[ ]:

print('Loading the model')
load_model = tensorflow.keras.models.load_model("motionsense_my_encoder4096.h5")
encoded_data = load_model.predict(test_data)


# In[ ]:


predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)


# In[ ]:


print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
print(metrics.classification_report(y_test, predict, labels=test_activities))


# In[ ]:


confusion_Matrix(y_test, predict, test_activities, " autoencoders")


# In[ ]:




