#!/usr/bin/env python
# coding: utf-8

# ## Working code

# In[ ]:


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


# In[ ]:


root = 'datasets/motion-sense-master/data/'
window_size = 350
time_step  = 10


# In[ ]:


def min_max_scaler(df):
#     return (df-df.min(axis=0))/(df.max(axis=0)-df.min(axis=0))
#     return df/df.max(axis=0)
    return (df-df.mean(axis=0))/df.std(axis=0) 


def convert_2D_3D(features, label):
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
    Y_local = Y_seq
    X_sequence = np.array(X_local)
    Y = np.array(Y_local)
    return X_sequence, Y


# In[ ]:


#Function to extract the data from the file

def motionsense_Extract(subset_data):
    
    #Function scope variables
    
    extracted_data = []
    extracted_label = []
    pullout_data = []
    pullout_label = []
    
       
    #Extracting only values
    print('\n Null data values in number of columns is {}' .format((subset_data.isnull().any().sum())))
    print('\n Null data values in number of rows is {}'  .format((subset_data.isnull().any(axis=1).sum())))
#     subset_data.dropna(axis=0, how='any', inplace=True)
    print('\n Null data values in number of columns is {}' .format((subset_data.isnull().any().sum())))
    print('\n Null data values in number of rows is {}'  .format((subset_data.isnull().any(axis=1).sum())))

    subset_data['attitude.pitch'] = min_max_scaler(subset_data['attitude.pitch'])
    subset_data['attitude.yaw'] = min_max_scaler(subset_data['attitude.yaw'])
    subset_data['attitude.roll'] = min_max_scaler(subset_data['attitude.roll'])
    subset_data['gravity.x'] = min_max_scaler(subset_data['gravity.x'])
    subset_data['gravity.y'] = min_max_scaler(subset_data['gravity.y'])
    subset_data['gravity.z'] = min_max_scaler(subset_data['gravity.z'])
    subset_data['rotationRate.x'] = min_max_scaler(subset_data['rotationRate.x'])
    subset_data['rotationRate.y'] = min_max_scaler(subset_data['rotationRate.y'])
    subset_data['rotationRate.z'] = min_max_scaler(subset_data['rotationRate.z'])
    subset_data['userAcceleration.x'] = min_max_scaler(subset_data['userAcceleration.x'])
    subset_data['userAcceleration.y'] = min_max_scaler(subset_data['userAcceleration.y'])
    subset_data['userAcceleration.z'] = min_max_scaler(subset_data['userAcceleration.z'])
    
    
    data_sliced = []
    slice_labels = []
    max_label = ' '

    #Sliding the window over the extracted data
    for i in range(0, len(subset_data) - window_size, time_step):
        ax = subset_data['attitude.pitch'].values[i: i + window_size]
        ay = subset_data['attitude.yaw'].values[i: i + window_size]    
        az = subset_data['attitude.roll'].values[i: i + window_size]
        gx = subset_data['gravity.x'].values[i: i + window_size]
        gy = subset_data['gravity.y'].values[i: i + window_size]
        gz = subset_data['gravity.z'].values[i: i + window_size]
        rx = subset_data['rotationRate.x'].values[i: i + window_size]
        ry = subset_data['rotationRate.y'].values[i: i + window_size]
        rz = subset_data['rotationRate.z'].values[i: i + window_size]
        ux = subset_data['userAcceleration.x'].values[i: i + window_size]
        uy = subset_data['userAcceleration.y'].values[i: i + window_size]
        uz = subset_data['userAcceleration.z'].values[i: i + window_size]    
        
              

        data_sliced.append([ax, ay, az, gx, gy, gz, rx, ry, rz, ux, uy, uz])
        max_label = stats.mode(subset_data['act'][i: i + window_size])[0][0]
        slice_labels.append(max_label)

    #Reshape the data to match the original data shape (n X Window_size X 3)
    #and store data and labels in the list
    data_sliced = np.asarray(data_sliced, dtype=np.float32).transpose(0, 2, 1)
    pullout_data.append(data_sliced)
    pullout_label.append(slice_labels)

    #Break the list of list for further operations
    extracted_data = [i for data in pullout_data for i in data]
    extracted_label = [i for label in pullout_label for i in label]

        
    
    return extracted_data, extracted_label


# In[ ]:


def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv(root+"data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = root +'A_DeviceMotion_data/'+'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
#________________________________


ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS [0:6]
print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    
dataset.tail()


# In[ ]:


dataset['act'] = dataset['act'].map({0.0:'Downstairs',
                                     1.0:'Upstairs',
                                     2.0:'Walking',
                                     3.0:'Jogging',
                                     4.0:'Standing',
                                     5.0:'Sitting'})


# In[ ]:


extracted_data, extracted_label = motionsense_Extract(dataset)
test_data = np.asarray(extracted_data)
print("Shape of the data after sliding window")
print(test_data.shape)
# extracted_data[:10]

#with open("motionsense_data_"+str(time_step)+"_"+str(window_size)+".pkl", "rb") as f:
#    test_data = np.asarray(pickle.loads(f.read()))

#with open("motionsense_label_"+str(time_step)+"_"+str(window_size)+".pkl", "rb") as f:
#    label = np.asarray(pickle.loads(f.read()))

#print("Shape of the data after sliding window")
#print(test_data.shape)

# In[ ]:


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
    max_amplitude = max(amplitude_list)
    max_index = amplitude_list.index(max_amplitude)
    max_phase = phase_list[max_index]
        
#     return (restored_sig + p[0] * t, max_amplitude, max_phase)
    return (restored_sig + p[0] * t, max_amplitude)


# Function for extracting amplitude and phase
def DCT(data, n_predict):
    
    amplitude_list = []
    phase_list = []
    n = data.size
    
    # number of harmonics in model
    n_harm = 8                     
    t = np.arange(0, n)
    p = np.polyfit(t, data, 1) 
    
    # find linear trend in x, detrended x in the frequency domain
    data_notrend = data - p[0] * t        
    data_freqdom = fftpack.dct(data_notrend) 
    
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
    max_amplitude = max(amplitude_list)
    max_index = amplitude_list.index(max_amplitude)
    max_phase = phase_list[max_index]
        
#     return (restored_sig + p[0] * t, max_amplitude, max_phase)
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
#     fft_data = np.zeros((data.shape[0], data.shape[-1]*2))
    fft_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[-1]): 
            
            _, fft_data[i][j] = FFT(data[i][0:,j], 0)
#             _, fft_data[i][1] = FFT(data[i][0:,1], 0)
#             _, fft_data[i][2] = FFT(data[i][0:,2], 0)
            
            
#         _, fft_data[i][0], fft_data[i][1] = FFT(data[i][0:,0], 0)
#         _, fft_data[i][2], fft_data[i][3] = FFT(data[i][0:,1], 0)
#         _, fft_data[i][4], fft_data[i][5] = FFT(data[i][0:,2], 0)

               
    return fft_data
# Function to call the FFT feature extraction function
def feature_DCT(data):
#     fft_data = np.zeros((data.shape[0], data.shape[-1]*2))
    dct_data = np.zeros((data.shape[0], data.shape[-1]))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[-1]): 
            
            _, dct_data[i][j] = DCT(data[i][0:,j], 0)
#             _, fft_data[i][1] = FFT(data[i][0:,1], 0)
#             _, fft_data[i][2] = FFT(data[i][0:,2], 0)
            
            
#         _, fft_data[i][0], fft_data[i][1] = FFT(data[i][0:,0], 0)
#         _, fft_data[i][2], fft_data[i][3] = FFT(data[i][0:,1], 0)
#         _, fft_data[i][4], fft_data[i][5] = FFT(data[i][0:,2], 0)

               
    return dct_data
    


# In[ ]:


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


#Function calls for all the features for different files
# print("Extracting Features")

# mean_test_data = feature_Mean(test_data)
# print("Mean calculated")


# std_test_data  = feature_Standard_Deviation(test_data)
# print("Standard Deviation calculated")


# var_test_data = feature_Variance(test_data)
# print("Variance calculated")


# ent_test_data = feature_Entropy(test_data)
# print("Entropy calculated")


# abdev_test_data = feature_Median_Absolute_Deviation(test_data)
# print("Absolute Deviation calculated")

# maxpeak_test_data = feature_Max_Peak(test_data)
# print("Max peak calculated")


# minpeak_test_data = feature_Min_Peak(test_data)
# print("Min peak calculated")


# cor_test_data = feature_correlation(np.nan_to_num(test_data))
# print("Correlation calculated")


# fft_test_data = feature_FFT(test_data)
# print("FFT calculated")

# dct_test_data = feature_DCT(test_data)
# print("DCT calculated")


# In[ ]:


# with open("motionsense_fft_data_"+str(time_step)+"_"+str(window_size)+".pkl", "wb") as f:
#     f.write(pickle.dumps(fft_test_data))
# with open("motionsense_dct_data_"+str(time_step)+"_"+str(window_size)+".pkl", "wb") as f:
#     f.write(pickle.dumps(dct_test_data))


# In[ ]:


# test_features = np.hstack((mean_test_data, std_test_data, var_test_data,
#                           cor_test_data, abdev_test_data, maxpeak_test_data, 
#                           minpeak_test_data, fft_test_data ))


# label = np.asarray(extracted_label)

# predict, y_test = classification_Using_SVM(np.nan_to_num(test_features), label)


# In[ ]:


#with open("motionsense_data_"+str(time_step)+"_"+str(window_size)+".pkl", "wb") as f:
#    f.write(pickle.dumps(extracted_data))
#with open("motionsense_label_"+str(time_step)+"_"+str(window_size)+".pkl", "wb") as f:
#    f.write(pickle.dumps(extracted_label))


# In[ ]:


test_activities = ['Walking', 'Jogging', 'Sitting',
                   'Standing', 'Upstairs', 'Downstairs']


# In[ ]:


label = np.asarray(extracted_label)


# In[ ]:


# print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
# print(metrics.classification_report(y_test, predict, labels=test_activities))
# confusion_Matrix(y_test, predict, test_activities, " Feature-based method")


# In[ ]:


def autoencoder(total_data, activation_fn, num_features, epoch, batch_n):
    input_layer = Input(shape=(total_data.shape[1],total_data.shape[2], ))
    encoder = LSTM(num_features, activation=activation_fn, kernel_initializer="he_uniform")(input_layer)
    #encoder = LSTM(180, activation='sigmoid')(encoder)
    decoder = RepeatVector(total_data.shape[1])(encoder)
    #decoder = LSTM(96, return_sequences=True, 
    #               activation='sigmoid')(decoder)
    decoder = LSTM(total_data.shape[2], return_sequences=True, 
                   activation=activation_fn,kernel_initializer="he_uniform")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
#     output = TimeDistributed(Dense(total_data.shape[2]))(decoder) 
    
#     autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.summary()
    encoderModel = Model(input_layer, encoder)
    autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
    autoencoder.fit(total_data, total_data, epochs = epoch, batch_size = batch_n, validation_split=0.2, verbose=1)
    #print("Saving the model")
    encoded_data = encoderModel.predict(total_data)
    #autoencoder.save("motionsense_my_autoencoder"+str(batch_n)+".h5")
    #encoderModel.save("motionsense_my_encoder"+str(batch_n)+".h5")
    return encoded_data


def autoencoder2l(total_data, activation_fn, num_features, epoch, batch_n):
    input_layer = Input(shape=(total_data.shape[1],total_data.shape[2], ))
    encoder = LSTM(6, return_sequences= True, activation=activation_fn, kernel_initializer="he_uniform")(input_layer)
    encoder = LSTM(30, activation=activation_fn)(encoder)
    decoder = RepeatVector(total_data.shape[1])(encoder)
    decoder = LSTM(30, return_sequences=True, activation=activation_fn, kernel_initializer="he_uniform")(decoder)
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
# maxnum_feature=300
# accuracy = np.zeros(maxnum_feature)
# loss = np.zeros(maxnum_feature)
# for i in range(1, maxnum_feature+1):
#     encoded_data = autoencoder(test_data, 'sigmoid', i, 1, 32)
#     print("Done calculated for : {}".format(i))
#     predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
#     accuracy[i] = metrics.accuracy_score(y_test, predict)
#     loss[i] = 100-accuracy[i]
#     with open("motionsense_accuracy_he.pkl", "wb") as f:
#         f.write(pickle.dumps(accuracy))
#     with open("motionsense_loss_he.pkl", "wb") as f:
#         f.write(pickle.dumps(loss))

# encoded_data = autoencoder(test_data, 'sigmoid', 180, 10, 64)
# predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
# print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
# print(metrics.classification_report(y_test, predict, labels=test_activities))
# confusion_Matrix(y_test, predict, test_activities, " autoencoders 64")
# # In[ ]:

# encoded_data = autoencoder(test_data, 'sigmoid', 180, 10, 32)
# predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
# print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
# print(metrics.classification_report(y_test, predict, labels=test_activities))
# confusion_Matrix(y_test, predict, test_activities, " autoencoders 32")

# # In[ ]:

# encoded_data = autoencoder(test_data, 'sigmoid', 180, 10, 16)
# predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
# print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
# print(metrics.classification_report(y_test, predict, labels=test_activities))
# confusion_Matrix(y_test, predict, test_activities, " autoencoders 16")

# # In[ ]:


# encoded_data = autoencoder(test_data, 'sigmoid', 180, 10, 8)
# predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
# print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))
# print(metrics.classification_report(y_test, predict, labels=test_activities))
# confusion_Matrix(y_test, predict, test_activities, " autoencoders 8")

# In[ ]:

print("sigmoid function-180,20")
encoded_data = autoencoder(test_data, 'sigmoid', 180, 20, 8)
predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))


#print("relu function")
#encoded_data = autoencoder(test_data, 'relu', 180, 1, 64)
#predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
#print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))

#print("tanh function")
#encoded_data = autoencoder(test_data, 'tanh', 180, 1, 64)
#predict, y_test = classification_Using_SVM(np.nan_to_num(encoded_data), label)
#print('Accuracy score: {}'.format(metrics.accuracy_score(y_test, predict)))

print('Model trained')

