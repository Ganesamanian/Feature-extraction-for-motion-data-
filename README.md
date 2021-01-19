# Feature-Extraction-for-Motion-Data-

This repository contains a implementation of Autoencoders and FFT for feature extraction from motion data.


# Abstract

Motion data is the readings from the sensors used to detect movement. These sensor
readings are time-series data. Time-series data is discrete values ordered sequentially
according to the time; one example of this, the stock price of Amazon. Smartphones and
smartwatches used everyday consist of sensors such as accelerometers and gyroscopes to
detect movement. The time-series data from these sensors collectively called motion data
and is used for Human Activity Recognition or Classification (HAR/HAC).

Motion data is challenging to interpret due to its higher dimensions and complexity.
Extracting features from the motion data, then utilizing these features for HAR/HAC
overcomes the challenges. Current State-Of-The-Art (SOTA) methods defined in the
literature either transform the time-series data to images or a different domain. Besides,
some methods are neither reusable nor time-efficient as regards processing extensive data.

Motion data can be broken down into many small data series, and the features from
those small series can be independently extracted. This research work uses a sliding win-
dow for the disintegration process and autoencoders for feature extraction. Additionally,
feature-based methods from SOTA are separately implemented to validate the autoen-
coders’ performance. The feature-based methods consist of a Fast Fourier Transform
(FFT) and several statistical features. The extracted features from autoencoders and
feature-based methods are individually provided as input to the Support Vector Machine
(SVM) for HAC. The SVMs’ accuracy indicates the significance of the method used to
extract features from motion data.

This proposed method has experimented with three annotated HAC datasets, namely
Motionsense (not the sensor manufacturer), Wireless Sensor Data Mining (WISDM),
and Wireless Sensor Data Mining version 1.1 (WISDM v 1.1) datasets. The accuracy of
the feature-based methods on the Motionsense dataset is 99.38%, the WISDM dataset
is 99.65%, and the WISDM v 1.1 is 85.50%. The autoencoders’ accuracy is 98.64%,
99.99%, and 70.46%, respectively. The autoencoders’ performance is competitive with
feature-based methods. Thus, this research has proven that autoencoders can extract
features from motion data without transforming the raw data. In addition to this, the
autoencoders have the further advantage of supporting code reusability and processing
the extensive datasets.

# Software Requirements 

To replicate this research work following software packages are necessary
1. Python3 >= 3.5.X
2. Numpy >= 1.19.X
3. Pandas >= 1.1.5
4. Seaborn >= 0.11.1
5. Pickle >= 5.0.0
6. Glob2 >= 0.7.X
7. Matplotlib >= 3.3.2
8. Scipy >= 1.5.2
9. Sklearn/ Scikit-learn >= 0.23.2
10. Tensorflow gpu >= 2.3.X
11. Keras >= 2.4.3
12. Jupyter Notebook >= 5.4.1
13. Anaconda >= 3.0.0 (optional)


It is advisable to use Anaconda that makes a constraint environment. The script
for this research work is written in Python following the Python3 format. Libraries
like Numpy, Scipy are used for computation and feature extraction in the feature-based
methods. Keras with TensorFlow framework is used to build the deep learning model
and training. Seaborn provides better visualization along with Matplotlib. The feature
array is large (219601, 180), so it should be stored in a file for visualization and analysis.
Pickle helps to store the data in a .pkl format with less consumption of memory. Glob2
reads all the file names in the specified directory and format. Pandas are utilized to read
the data from the files stored by Glob2. Scikit learn is used to provide a classification
report, train the SVM model as well as to divide the features for training and testing.



## How to execute this
Clone this into your local machine:
'''
git clone git@github.com:Ganesamanian/Feature-extraction-for-motion-data-.git
'''

The repository includes
1. Dataset directory contains the three datasets.
2. Script directory contains the script files to execute the code and load file to run
the saved model.
3. Model directory contains the saved models.
4. Result directory contains experiment results and .pkl files.


### Instructions:
###### Step 1(Script):
For each dataset the script files are kept sepearate. So based on the dataset choose the required script, E.g:- HAR_motionsense.py 

###### Step 2(Path):
Kindly change the root path according to the cloned folder location in your system. Folder path and file path remains the same

###### Step 3(Run):
Now its good to run using the command python3 HAR_motionsense.py from terminal inside the script folder. 


####### Disclaimer:
The project requires a computational resource equivalent and above to the Nvidia GeForce
GTX 1650 graphics card with 4GB of video Random Access Memory (RAM) for extracting the features.
Check the machine configuration before executing the code.









