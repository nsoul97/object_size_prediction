# Dataset Description

## Participants

The dataset's movements were performed by **8 participants**:
- P1
- P2
- P3
- P4
- P5
- P6
- P7
- P8

All the movements that were performed by a participant are listed in the corresponding directory. <br/>
e.g. All the movements that were performed by the P4 participant are listed in the P4 directory. <br/><br/>

## Objects

Each participant reached for **3 objects**:
* a small-sized cube with a 2.5 cm edge length (S)
* a medium-sized cube with a 5.5 cm edge lenth (M)    
* a large-sized cube with a 7.5 cm edge length (L) 

<p align="center">
<img src="objects.png" width=400></img>
</p>

The blue cube is the small object, the red cube is the medium object and the green cube is the large object. <br/> <br/>

## Total Recorded Movements

Each participant reached for each of the small, the medium and the large object 30 times. Therefore:
- 30 movements were recorded for each (participant, object) pair,
- 90 movements were recorded for each participant and
- 720 movements were recorded in total. <br/><br/>

## Movement Identification

A movement can be identified based on:
- the participant's identifier (P1, P2, P3, P4, P5, P6, P7, P8)
- the size of the target object (S, M, L)
- the numerical identifier of the movement for this (participant, object) pair  (01, 02, ..., 29, 30)

e.g. P1_S_12: The movement was performed by the P1 participant. In this movement the participant reached for the small object (S). Out of the 30 total recorded movements of the (P1,S) pair, this is the 12th movement. <br/><br/>

## Final dataset

Out of these 720 movements, 5 movements where the data collection was problematic were identified and excluded from the dataset. Specifically, the following files were excluded from the dataset:
* P2_S_15.csv
* P4_M_07.csv
* P5_L_03.csv
* P8_M_03.csv
* P8_L_01.csv

As a result, the final dataset consists of **715 movements**. <br/><br/>

## Skeletal data

Each file of the dataset contains the skeletal data of the participant. The skeletal data was extracted using the **OpenPose** framework. The OpenPose framework was used to estimate the 2D locations of 18 body joints and 42 hand joints (21 hand joints for each of the left and the right hand). Each file contains only the joints that could potentially be helpful for predicting the object-to-be-grasped. Specifically, each file contains:
* the right shoulder (RShoulder), the right elbow (RElbow) and the right wrist (RWrist) full body joints
* the right hand joints

Therefore, 24 joints are listed in total in each csv file:
* 3 full body joints
* 21 right hand joints
<br/><br/>

<p align="center">
<img src="openpose_keypoints.png"></img>
</p>

The following names are used to refer to the corresponding joints of the images above:

| OpenPose Joint Enumeration | Joint Name          | 
| -------------------------- | ------------------- |
| Full Body -  5             | RShoulder           |
| Full Body -  6             | RElbow              |
| Full Body -  7             | RWrist              |
| Right Hand - 0             | RPalmBase           |
| Right Hand - 1             | RThumb1CMC          |
| Right Hand - 2             | RThumb2Knuckles     |
| Right Hand - 3             | RThumb3IP           |
| Right Hand - 4             | RThumb4FingerTip    |
| Right Hand - 5             | RIndex1Knuckles     |
| Right Hand - 6             | RIndex2PIP          |
| Right Hand - 7             | RIndex3DIP          |
| Right Hand - 8             | RIndex4FingerTip    |
| Right Hand - 9             | RMiddle1Knuckles    |
| Right Hand - 10            | RMiddle2PIP         |
| Right Hand - 11            | RMiddle3DIP         |
| Right Hand - 12            | RMiddle4FingerTip   |
| Right Hand - 13            | RRing1Knuckles      |
| Right Hand - 14            | RRing2PIP           |
| Right Hand - 15            | RRing3DIP           |
| Right Hand - 16            | RRing4FingerTip     |
| Right Hand - 17            | RPinky1Knuckles     |
| Right Hand - 18            | RPinky2PIP          |
| Right Hand - 19            | RPinky3DIP          |
| Right Hand - 20            | RPinky4FingerTip    |
<br/>

Apart from the **x** and **y** coordinates of a joint, OpenPose also outputs a probability (**prob**) as the confidence value of the estimated location of the joint. For each of the joint detections, the x- and y-coordinates and the probability are listed in the csv files. In case OpenPose fails to predict the position of a joint for a frame, the x,y and prob values of the joint are set to 0 for this frame. <br/>
e.g. the RShoulder.x column contains the x-coordinates of the RShoulder joint for each frame of the recorded movement.<br/>

Finally, each file contains a "Time" column. In this column, the elapsed time from the first recorded frame of the movement is listed for each frame. For the first frame, the time is always set to 0. The elapsed time is measured in seconds. <br/><br/>

In conclusion, each file contains 73 columns:
- 3 columns (x,y,prob) for each of the 24 joints
- an extra "Time" column

Each row corresponds to a frame of the recorded movement. 