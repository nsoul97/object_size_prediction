import os
import pandas as pd
import re
import random
import numpy as np

def read_dataset(path="dataset/data/", joints=None, attr=None):
    """Read all the movements of the dataset and store them in a dictionary. Choose which joints to read for each movements and which attributes to read for each joint.

    Args:
        path (str, optional): The path of the movement dataset. Defaults to "dataset/data/".
        joints (list, optional): The joints (RShoulder, RElbow, ..., RPinky4FingerTip) of interest for all the movements. Defaults to None. In this case all 24 joints are used for all the movements.
        attr (list, optional): The attributes (x,y,prob) of interest for all the joints. Defaults to None. In this case all 3 attributes are used for all the joints.

    Returns:
        data (dictionary): A dictionary with the movement filename as the key and a pandas dataframe containing the corresponding skeletal data of the movement as the value.
    """
    if joints:
        if attr is None:
            attr = ['x','y','prob']
        cols = ["Time"] + [f"{j}.{at}" for j in joints for at in attr]
    else:
        cols = None
    data = dict()
    participants = os.listdir(path)
    for p in participants:
        movements = os.listdir(os.path.join(path,p))
        for m in movements:
            if not re.match(f'{p}_[S|M|L]_(0[1-9]|[1-2][0-9]|30).csv',m): continue
            df = pd.read_csv(os.path.join(path,p,m), usecols=cols)
            data[m]=df
    return data


def clean_data(mov_data):
    """Identify the frames with noisy right wrist y-coordinates' detections and filter them out.

    Args:
        mov_data (pd.DataFrame): A DataFrame containing the skeletal data of a single movement of the dataset, as given by OpenPose.

    Returns:
        mov_data (pd.DataFrame): A DataFrame containing the skeletal data of the given movement with frames that have noisy wrist y-coordinates filtered out.
    """
    
    PROB_THRESHOLD = 0.6
    DIST_THRESHOLD = 10

    mov_data = mov_data.loc[mov_data["RWrist.prob"] > PROB_THRESHOLD]                       # Remove the frames where the detection probability of the right wrist keypoint is less than or equal to 0.6
    
    coords = mov_data["RWrist.y"].to_numpy()                                                # Calculate the minimum distance between the right wrist keypoint of the current and previous frame and the current and next frame. 
    dist = np.abs(coords[1:] - coords[:-1])
    prev_dist = np.r_[dist[0], dist]
    next_dist = np.r_[dist, dist[-1]]
    neighbour_dist = np.min([prev_dist, next_dist], axis=0)
  
    mov_data = mov_data[neighbour_dist < DIST_THRESHOLD]                                    # If the distance for a frame is greater than or equal to 10 pixels, the frame is filtered out.

    return mov_data


def cut_movement(cleaned_data, mov_data, tail=0, head=0):
    """Identify and isolate the data that correspond to the grasping phase of the movement and the data that correspond to the given number of frames before (tail) and after (head) the grasping phase.

    Args:
        cleaned_data (pd.DataFrame): A DataFrame containing the skeletal data of a single movement of the dataset with the frames with noisy wrist y-coordinates filtered out.
        mov_data (pd.DataFrame): A DataFrame containing the skeletal data of the same movement of the dataset, as given by OpenPose.
        tail (int, optional): An integer that indicates how many conseutive frames before the grasping phase to return. If negative or not given, the value defaults to 0.
        head (int, optional): An integer that indicates how many conseutive frames after the grasping phase to return. If negative or not given, the value defaults to 0.

    Returns:
        mov_data [pd.DataFrame]: A DataFrame containing the skeletal data of the grasping phase of the given movement and the skeletal data of the given number of frames before and after the 
                                 grasping movement.
    """
    
    WINDOW = 10
    START_STD = 2.0
    STOP_STD = 1.5

    coords = cleaned_data["RWrist.y"].to_numpy()

    wrist_std = []                                                                          # Calculate the standard deviation of the right wrist's y-coordinate for a window of 10 frames.
    for i in range(len(coords)):
        mean = np.average(coords[i:i+WINDOW])
        wrist_std.append(np.sqrt(np.average((coords[i:i+WINDOW]-mean)**2)))
    
    wrist_std = np.array(wrist_std)                                         
    peak = np.argmax(wrist_std)                                                             # Identify the peak of the standard deviation distribution.
    start_ind = np.where(wrist_std[:peak] > START_STD)[0]                                   # The grasping movement starts the first time the standard deviation becomes over 2.
    end_ind = np.where(wrist_std[peak:] < STOP_STD)[0]                                      # The grasping movement ends after the peak of the standard deviation, the first time the standard deviation becomes less than 1.5.

    start = np.min(np.where(wrist_std[:peak] > START_STD)) + WINDOW - 1                     # The standard deviation is assigned as a measure of speed of the last frame of the window.
    end = peak + np.min(np.where(wrist_std[peak:] < STOP_STD)) + WINDOW -1

    start_time, end_time = cleaned_data.iloc[[start,end]]["Time"]

    start_ind = max(0, mov_data[mov_data["Time"] == start_time].index[0] - max(0,tail))     # Add the given number of frames before the grasping phase.
    end_ind = mov_data[mov_data["Time"] == end_time].index[0] + max(0, head)                # Add the given number of frames before the grasping phase.

    return mov_data.iloc[start_ind:end_ind]