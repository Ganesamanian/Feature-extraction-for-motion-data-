This dataset includes activity data from 51 participants from an activity
recognition project. Each participant performed each of the 18 activities 
(listed in activity_key.txt) for 3 minutes and the sensor data (accelerometer 
and gyroscope for smartphone and smartwatch) was recorded at a rate of 20 Hz. 
The smartphones used were Nexus 5, Nexus 5X, and Galaxy S6 while the smartwatch
used was the LG G watch. This dataset contains the raw data as well as the
transformed data which was generated using arffmagic (located in
arffmagic-master).

A much more detailed description of the two main data sets is provided in the
wisdm-dataset-description.pdf document at the top level of the data directory.

################################################################################
################################################################################

The raw sensor data is located in the raw directory. Each user has its own data
file which is tagged with their subject id, the sensor, and the device.  Within
the data file, each line is:

Subject-id, Activity Label, Timestamp, x, y, z

The features are defined as follows:

subject-id: Identfies the subject and is an integer value between 1600 and 1650.

activity-label: see activity_key.txt for a mapping from 18 characters to the
       activity name

timestamp: time that the reading was taken (Unix Time)

x: x sensor value (real valued) 
y: y sensor value (real valued) 
z: z sensor value (real valued) 

################################################################################
################################################################################

ARFF files for each user (with attributes described below) are located in the 
arff_files directory. The ARFF files are created using the arffmagic program, which
aggregates the lower level raw data into examples labeled with the activity being
performed. The descriptions of these features are also found in the many WISDM
activity recognition papers. 

Recognize that the sensors are samples at 20Hz and we use a 10 second window size,
so most of the statistics are based on the aggregation of 200 sensor readings.

Attribute information:
ACTIVITY[1]:
	This field contains the code that uniquely identifies the activity
	(activity_key.txt provides the mapping from code to actual activity)

Binned Distribution[30]:
	The range of values is determined (maximum - minimum), 10 equal-sized
	bins are formed, and the fraction of the 200 values within each bin
	is recorded for each of the 3 axes. The axis bin values are provided
	in the following order: x, y, z (10 values for each). The attributes are
	named X0 .. X9, Y1 ... Y9, and Z0 ... Z9

Average[3]:
	 Average sensor value (for each of the 3 axes). Labeled XAVG, YAVG, ZAVG


Time Between Peaks[3]:
	Time between peaks in the sinusoidal waves formed by the data
	as determined by a simple algorithm (feel free to check the code). Done
	for each axis). Labeled XPEAK, YPEAK, ZPEAK.

Average Absolute Difference[3]:
	Average absolute difference between the 200 values and the mean of these
	values (for each axis). Labeled as {X,Y,Z}ABSOLDEV

Standard Deviation[3]:
	 Standard deviation (for each axis). Labeled as {X,Y,Z}STANDDEV

Variance[3];
	The variance of the values (for each axis). Labeled as {X,Y,Z}VAR

--------------------------------------------------------------------------------
THE NEXT THREE SETS OF FEATURES ARE NOT USED IN OUR PUBLISHED RESEARCH PAPERS. THEY 
WERE ADDED TO EXPERIMENT WITH.  

MFCC: Mel-frequency cepstral coefficients [39]
	MFCCs are a representation of the short-term power spectrum of a wave, based
	on a linear cosine transform of a log power spectrum on a nonlinear mel scale
	of frequency. There are 13 per axis There are 13 per axis. They are labeled as
	XMFCC{0-12}, YMFCC{0-12}, and ZMFCC{0-12}. 


Cosine distance [3]:
	These are the cosine distances between the sensor values for a pair of axes.
	Three pairs are considered to cover all possible pairs. They are labeled as
	{XY, XZ, YZ}COS.

Correlation [3]:
	These are the correlations between the sensor values for a pair of axes.
	Three pairs are considered to cover all possible pairs. They are labeled as
	{XY, XZ, YZ}COR.

----------------------------------------------------------------------------------

Average Resultant Acceleration[1]:
	For each of the sensor samples in the window, take the square root of the sum
	of the square of the x, y, z axis values, and then average them. Labeled as
	RESULTANT.

class: Subject-id (this name is misleading since for activity recognition the class is
	the activity in the first position).  
