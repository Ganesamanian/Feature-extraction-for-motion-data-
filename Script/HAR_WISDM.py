#!/usr/bin/env python
# coding: utf-8

# 
# ### Created on Thursday  August 15, 2020
# 
# ### Author: Ganesamanian Kolappan
# 

# ## Importing libraries
# 

# ##### Importing the libraries to be used in the whole application, for the neatness of the code all the imports are done at the start of the coding.

# In[1]:


import numpy as np
from numpy import fft
import pandas as pd
import seaborn as sns
import pickle
import glob
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm


from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical, vis_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Input, RepeatVector


# ## Variable Declaration

# #### Some variables are decalred as global so that it can be used anywhere in the coding

# In[2]:


#Dataframe variables

column_names = [
    'user-id',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

labels = [
    'A', #Walking
    'B', #Jogging
    'C', #Climbing Stairs
    'D', #Sitting
    'E', #Standing
    'F'  #Typing
]

#Data path

root = '/home/ganesh/Documents/R&D/'

folder_path = 'datasets/wisdm-dataset/raw/'

#Class names
activities = ['Walking', 'Jogging', 'Climbing Stairs', 'Sitting', 'Standing', 'Typing',
              'Brushing Teeth', 'Eating Soup', 'Eating Chips', 'Eating Pasta',
              'Drinking from cup', 'Eating Sandwich', 'Kicking', 'Playing Tennis',
              'Basketball', 'Writing', 'Clapping', 'Folding Clothes']


#Windowing variables
time_step = 2
window_size = 200


# ## Function call- Data Preprocessing

# #### First the data is being read from the file and stored as pandas dataframe, then converting the class labels to class name after that sliding window approach is being applied in order to get the samples from the readings. This is done for all the 51 observation from the four sensors (acelerometer and gyroscope from phone and watch). In order to have the same size of data across the four device, there is small patch of code which limits the sample to minimum occurance so that the same number of samples is maintained through out.

# In[3]:


#Function to extract the data from the file

def data_Extraction(filename):
    
    #Function scope variables
    
    extracted_data = []
    extracted_label = []
    pullout_data = []
    pullout_label = []
    final_data = []
    final_label = []
    
    
    for filename in glob.glob(filename):

        subset_data = pd.read_csv(filename, header=None, names = column_names)
        #Extracting only values
        subset_data['z-axis'] = subset_data['z-axis'].str.replace(';', '').astype(float)

        #Detecting the end-point of sixth activity and ignoring
        #the rest of the 12 activities

#         idx = int(data.loc[data['activity'] == 'G'].index[0])

#         subset_data = pd.DataFrame(data[:idx])

        #Renaming the activities from alphabatical representation
        subset_data['activity'] = subset_data['activity'].map({'A':'Walking',
                                                               'B':'Jogging',
                                                               'C':'Climbing Stairs',
                                                               'D':'Sitting',
                                                               'E':'Standing',
                                                               'F':'Typing',
                                                               'G':'Brushing Teeth',
                                                               'H':'Eating Soup',
                                                               'I':'Eating Chips',
                                                               'J':'Eating Pasta',
                                                               'K':'Drinking from cup',
                                                               'L':'Eating Sandwich',
                                                               'M':'Kicking',
                                                               'O':'Playing Tennis',
                                                               'P':'Basketball',
                                                               'Q':'Writing',
                                                               'R':'Clapping',
                                                               'S':'Folding Clothes'})

        #Local variable for storing the data from
        #sliding window approach

        data_sliced = []
        slice_labels = []
        max_label = ' '

        #Sliding the window over the extracted data
        for i in range(0, len(subset_data) - window_size, time_step):
            x = subset_data['x-axis'].values[i: i + window_size]
            y = subset_data['y-axis'].values[i: i + window_size]    
            z = subset_data['z-axis'].values[i: i + window_size]   

            data_sliced.append([x, y, z])
            max_label = stats.mode(subset_data['activity'][i: i + window_size])[0][0]
            slice_labels.append(max_label)
        
        #Reshape the data to match the original data shape (n X Window_size X 3)
        #and store data and labels in the list
        data_sliced = np.asarray(data_sliced, dtype=np.float32).transpose(0, 2, 1)
        pullout_data.append(data_sliced)
        pullout_label.append(slice_labels)

        #Break the list of list for further operations
        extracted_data = [i for data in pullout_data for i in data]
        extracted_label = [i for label in pullout_label for i in label]

        #Converting labels to dataframe
        slice_dataframe = pd.DataFrame(extracted_label, columns=['activity'])
        
                
        #Extracting the first "limited" data from extracted data 
        #in order to have same number of data from every file
        limit = int(3500/time_step)
        for i in activities:
            
            occurance_idx = int(slice_dataframe.loc[slice_dataframe['activity'] == i].index[0])
            
            final_data.append(extracted_data[occurance_idx:occurance_idx + limit])
            final_label.append(extracted_label[occurance_idx:occurance_idx + limit])
    
    #Flattening the list
    data = [i for data in final_data for i in data]
    label = [i for label in final_label for i in label]
    
    return data, label

# Function for the scatter plot 

def visualization_scatter(x, y, title, x_name, y_name, color, label):
    
#     plt.figure(figsize=(15,10))    
    plt.scatter(x, y, c=color, label = label)
    plt.title(title, fontsize=25)
    plt.grid()
    plt.legend(fontsize=20)
    plt.xlabel(x_name, fontsize=20)
    plt.ylabel(y_name, fontsize=20)
    plt.xticks(fontsize= 17)
    plt.yticks(fontsize= 17)

# Function for the scatter plot 
def visualization_graph(x, y, title, x_name, y_name, color, label):
    
        
    plt.plot(x, y, c=color, label = label)
    plt.title(title, fontsize=25)
    plt.grid()
    plt.legend(fontsize= 20, bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel(x_name, fontsize=20)
    plt.ylabel(y_name, fontsize=20)
    plt.xticks(fontsize= 17)
    plt.yticks(fontsize= 17)


 # In[6]:

# ## Function call - Features

# #### Features are being extracted from the hand made program, the main features that to be concentrated in FFT apart form that mean, variance, standard deviation, entropy, max and min of the peak, correlation between xy, yz, zx is being extracted. From FFT we get two features that is amplitude and phase. 


# Function for extracting amplitude and phase
def FFT(data, n_predict):
    
    amplitude_list = []
    phase_list = []
    n = data.size
    
    # number of harmonics in model
    n_harm = 8                     
    t = np.arange(0, n)
    p = np.polyfit(t, data, 1) 
    
    # find linear trend in x, detrended x in the frequency domain
    data_notrend = data - p[0] * t        
    data_freqdom = fft.fft(data_notrend) 
    
    # frequencies
    f = fft.fftfreq(n)              
    indexes = list(range(n))   
    
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(data_freqdom[i]))
    indexes.reverse()
    j=1
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    
    for i in indexes[:1 + n_harm * 2]:
        
        #Getting amplitude and phase
        amplitude = np.absolute(data_freqdom[i]) / n         
        phase = np.angle(data_freqdom[i])                  
        a = amplitude * np.cos(2 * np.pi * f[i] * t + phase)
              
        if j%4==0:
            amplitude_list.append(amplitude)
            phase_list.append(phase)
            
        restored_sig += a
        j+=1
        
    #Getting the maximum amplitude and corresponding phase
    max_amplitude = min(amplitude_list)
    max_index = amplitude_list.index(max_amplitude)
    max_phase = phase_list[max_index]
        
    return (restored_sig + p[0] * t, max_amplitude)

# Function to extract the mean feature
def feature_Mean(data):
    mean_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        mean_data[i] = np.mean(data[i], axis=0)
        
    return mean_data

# Function to extract the standard devaition feature
def feature_Standard_Deviation(data):
    std_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        std_data[i] = np.std(data[i], axis=0)
        
    return std_data

# Function to extract the variance feature
def feature_Variance(data):
    var_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        var_data[i] = np.var(data[i], axis=0)
        
    return var_data

# Function to extract the entropy feature
def feature_Entropy(data):
    ent_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        ent_data[i] = stats.entropy(data[i], base=2, axis=0)
        
    return ent_data

# Function to extract the absolute deviation feature
def feature_Median_Absolute_Deviation(data):
    abdev_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        abdev_data[i] = stats.median_absolute_deviation(data[i], axis=0)
        
    return abdev_data

# Function to extract the maximum of the peaks feature
def feature_Max_Peak(data):
    maxpeak_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        maxpeak_data[i] = np.max(data[i], axis=0)
        
    return maxpeak_data

# Function to extract the minimum of the peaks feature
def feature_Min_Peak(data):
    minpeak_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        minpeak_data[i] = np.min(data[i], axis=0)
        
    return minpeak_data

# Function to extract the correlation feature
def feature_correlation(data):
    cor_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        cor_data[i][0] = stats.pearsonr(data[i][0:,0], data[i][0:,1])[0]
        cor_data[i][1] = stats.pearsonr(data[i][0:,1], data[i][0:,2])[0]
        cor_data[i][2] = stats.pearsonr(data[i][0:,2], data[i][0:,0])[0]
        
        
    return cor_data

# Function to call the FFT feature extraction function
def feature_FFT(data):
    fft_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        _, fft_data[i][0] = FFT(data[i][0:,0], 0)
        _, fft_data[i][1] = FFT(data[i][0:,1], 0)
        _, fft_data[i][2] = FFT(data[i][0:,2], 0)        
        
    return fft_data
    

# ## Classification using SVM

# #### SVM is used for classification to know how the extracted features perform

# ### Function call - SVM and Confusion Matrix

# In[9]:


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
    sns.heatmap(cf_matrix, annot=True, cmap='Reds', fmt='g', 
                xticklabels=activities, yticklabels=activities)
    plt.title("Confusion matrix for"+title, fontsize=25)
    plt.xlabel("Predicted activities", fontsize=20)
    plt.ylabel("Actual activities", fontsize=20)
    plt.xticks(fontsize= 15, rotation=90)
    plt.yticks(fontsize= 15, rotation=0)
    plt.savefig('WISDM_confusion_matrix_'+title+"_"+str(time_step)+"_"+str(window_size)+'.png',
                bbox_inches = 'tight')
    plt.show()

# In[ ]:


def autoencoder(total_data, activation_fn, num_features, epoch, batch_n):
    input_layer = Input(shape=(total_data.shape[1],total_data.shape[2], ))
    encoder = LSTM((num_features*2), activation=activation_fn, kernel_initializer="he_uniform")(input_layer)
    encoder = LSTM(num_features, activation=activation_fn, kernel_initializer="he_uniform")(encoder)
    decoder = RepeatVector(total_data.shape[1])(encoder)
    decoder = LSTM(num_features, return_sequences=True, 
                   activation=activation_fn, kernel_initializer="he_uniform")(decoder)
    decoder = LSTM((num_features*2), return_sequences=True, 
                   activation=activation_fn, kernel_initializer="he_uniform")(decoder)
#   autoencoder = Model(inputs=input_layer, outputs=decoder)
    output = TimeDistributed(Dense(total_data.shape[2]))(decoder) 
    
    autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.summary()
    encoderModel = Model(input_layer, encoder)
    autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
    autoencoder.fit(total_data, total_data, epochs = epoch, batch_size = batch_n, validation_split=0.2, verbose=1)
    encoded_data = encoderModel.predict(total_data)
    autoencoder.save("motionsense_my_autoencoder.h5")
    encoderModel.save("motionsense_my_encoder.h5")
    return encoded_data

# In[4]:


file_path_phone_accel = 'phone/accel/*.txt'
extracted_phone_accel_data, extracted_phone_accel_label = data_Extraction(folder_path + file_path_phone_accel)
file_path_phone_gyro = 'phone/gyro/*.txt'
extracted_phone_gyro_data, extracted_phone_gyro_label = data_Extraction(folder_path + file_path_phone_gyro)
file_path_watch_accel = 'watch/accel/*.txt'
extracted_watch_accel_data, extracted_watch_accel_label = data_Extraction(folder_path + file_path_watch_accel)
file_path_watch_gyro = 'watch/gyro/*.txt'
extracted_watch_gyro_data, extracted_watch_gyro_label = data_Extraction(folder_path + file_path_watch_gyro)


# In[5]:


#Printing the shape to visualize the size is same

print(np.asarray(extracted_phone_gyro_data).shape)
print(np.asarray(extracted_phone_accel_data).shape)
print(np.asarray(extracted_watch_gyro_data).shape)
print(np.asarray(extracted_watch_accel_data).shape)



# In[7]:


#Function calls for all the features for different files

mean_phone_accel_data = feature_Mean(np.asarray(extracted_phone_accel_data))
mean_phone_gyro_data = feature_Mean(np.asarray(extracted_phone_gyro_data))
mean_watch_accel_data = feature_Mean(np.asarray(extracted_watch_accel_data))
mean_watch_gyro_data = feature_Mean(np.asarray(extracted_watch_gyro_data))

std_phone_accel_data  = feature_Standard_Deviation(np.asarray(extracted_phone_accel_data))
std_phone_gyro_data  = feature_Standard_Deviation(np.asarray(extracted_phone_gyro_data))
std_watch_accel_data  = feature_Standard_Deviation(np.asarray(extracted_watch_accel_data))
std_watch_gyro_data  = feature_Standard_Deviation(np.asarray(extracted_watch_gyro_data))

var_phone_accel_data = feature_Variance(np.asarray(extracted_phone_accel_data))
var_phone_gyro_data = feature_Variance(np.asarray(extracted_phone_gyro_data))
var_watch_accel_data = feature_Variance(np.asarray(extracted_watch_accel_data))
var_watch_gyro_data = feature_Variance(np.asarray(extracted_watch_gyro_data))

ent_phone_accel_data = feature_Entropy(np.asarray(extracted_phone_accel_data))
ent_phone_gyro_data = feature_Entropy(np.asarray(extracted_phone_gyro_data))
ent_watch_accel_data = feature_Entropy(np.asarray(extracted_watch_accel_data))
ent_watch_gyro_data = feature_Entropy(np.asarray(extracted_watch_gyro_data))

abdev_phone_accel_data = feature_Median_Absolute_Deviation(np.asarray(extracted_phone_accel_data))
abdev_phone_gyro_data = feature_Median_Absolute_Deviation(np.asarray(extracted_phone_gyro_data))
abdev_watch_accel_data = feature_Median_Absolute_Deviation(np.asarray(extracted_watch_accel_data))
abdev_watch_gyro_data = feature_Median_Absolute_Deviation(np.asarray(extracted_watch_gyro_data))

maxpeak_phone_accel_data = feature_Max_Peak(np.asarray(extracted_phone_accel_data))
maxpeak_phone_gyro_data = feature_Max_Peak(np.asarray(extracted_phone_gyro_data))
maxpeak_watch_accel_data = feature_Max_Peak(np.asarray(extracted_watch_accel_data))
maxpeak_watch_gyro_data = feature_Max_Peak(np.asarray(extracted_watch_gyro_data))

minpeak_phone_accel_data = feature_Min_Peak(np.asarray(extracted_phone_accel_data))
minpeak_phone_gyro_data = feature_Min_Peak(np.asarray(extracted_phone_gyro_data))
minpeak_watch_accel_data = feature_Min_Peak(np.asarray(extracted_watch_accel_data))
minpeak_watch_gyro_data = feature_Min_Peak(np.asarray(extracted_watch_gyro_data))

cor_phone_accel_data = feature_correlation(np.asarray(extracted_phone_accel_data))
cor_phone_gyro_data = feature_correlation(np.asarray(extracted_phone_gyro_data))
cor_watch_accel_data = feature_correlation(np.asarray(extracted_watch_accel_data))
cor_watch_gyro_data = feature_correlation(np.asarray(extracted_watch_gyro_data))

fft_phone_accel_data = feature_FFT(np.asarray(extracted_phone_accel_data))
fft_phone_gyro_data = feature_FFT(np.asarray(extracted_phone_gyro_data))
fft_watch_accel_data = feature_FFT(np.asarray(extracted_watch_accel_data))
fft_watch_gyro_data = feature_FFT(np.asarray(extracted_watch_gyro_data))
   


# In[8]:


#Concatenating the four files based on the feature group

mean_feature = np.hstack((mean_phone_accel_data, mean_phone_gyro_data, mean_watch_accel_data, mean_watch_gyro_data))
variance_feature = np.hstack((var_phone_accel_data, var_phone_gyro_data, var_watch_accel_data, var_watch_gyro_data))
std_feature = np.hstack((std_phone_accel_data, std_phone_gyro_data, std_watch_accel_data, std_watch_gyro_data))
entropy_feature = np.hstack((ent_phone_accel_data, ent_phone_gyro_data, ent_watch_accel_data, ent_watch_gyro_data))
abdev_feature = np.hstack((abdev_phone_accel_data, abdev_phone_gyro_data, abdev_watch_accel_data, abdev_watch_gyro_data))
maxpeak_feature = np.hstack((maxpeak_phone_accel_data, maxpeak_phone_gyro_data, maxpeak_watch_accel_data, maxpeak_watch_gyro_data))
minpeak_feature = np.hstack((minpeak_phone_accel_data, minpeak_phone_gyro_data, minpeak_watch_accel_data, minpeak_watch_gyro_data))
correlation_feature = np.hstack((cor_phone_accel_data, cor_phone_gyro_data, cor_watch_accel_data, cor_watch_gyro_data))
fft_feature = np.hstack((fft_phone_accel_data, fft_phone_gyro_data, fft_watch_accel_data, fft_watch_gyro_data))



# ## All the Features

# #### Classification performed by concatenating all the features

# In[10]:


# Concatenating all the features

features = np.hstack((mean_feature, variance_feature, np.nan_to_num(entropy_feature),
                      abdev_feature, maxpeak_feature, minpeak_feature, std_feature,
                      correlation_feature, fft_feature))
print(features.shape)

#Class label
label = np.asarray(extracted_phone_accel_label)

predict, y_test = classification_Using_SVM(features, label)

print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
print(metrics.classification_report(y_test, predict, labels=activities))


# In[11]:


confusion_Matrix(y_test, predict, activities, " Feature-based method")


# #### Accuracy is 5%  from all features since the entropy contains huge values, this makes the classifier to learn incorrectly. So the entropy feature doesn't hold nay good.

# ## All Features except entropy

# #### Classification performed by concatenating all the features except entropy since it doesn't hold any good

# In[12]:


features = np.hstack((mean_feature, variance_feature, std_feature,
                      abdev_feature, maxpeak_feature, minpeak_feature,
                      correlation_feature, fft_feature))

label = np.asarray(extracted_phone_accel_label)

predict, y_test = classification_Using_SVM(features, label)

print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
print(metrics.classification_report(y_test, predict, labels=activities))


# In[14]:


confusion_Matrix(y_test, predict, activities, " Feature-based method")


# ## Autoencoder

# In[ ]:


total_data = np.concatenate((extracted_phone_accel_data, 
                    extracted_phone_gyro_data,
                    extracted_watch_accel_data,
                    extracted_watch_gyro_data),
                   axis=2)


# In[ ]:


label = np.asarray(extracted_phone_accel_label)
encoded_data = autoencoder(total_data, 'sigmoid', 180, 20, 8)

predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)


# In[ ]:


print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
print(metrics.classification_report(y_test, predict, labels=activities))


# In[ ]:


confusion_Matrix(y_test, predict, activities, " autoencoders")

