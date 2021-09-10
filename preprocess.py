import os
import pandas as pd
import re
import random

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
            df = pd.read_csv(os.path.join(path,p,m), index_col="Time", usecols=cols)
            data[m]=df
    return data
            

def all_in_kfold(data):
    """Split the movements into 10 partitions of approximately equal size (+-1) according to the "all-in" strategy. The movements of each partition are selected randomly.
       The partitions have approximately the same number of movements (+-1) for a given (participant, object) pair.

    Args:
        data (dictionary): A dictionary with the movement filename as the key and a pandas dataframe containing the corresponding skeletal data of the movement as the value.

    Returns:
        partitions (list): A list containing 10 lists, one for each partition. The nested lists contain the movement filenames of the partition.
    """
    
    part_movements = {f"P{p}":{'S':[], 'M':[], 'L':[]} for p in range(1,9)}
    for mov in data.keys():
        part_id = mov[:2]
        obj_id = mov[3:4]
        part_movements[part_id][obj_id].append(mov)
    
    partitions = [[] for p in range(10)]
    for pfiles in part_movements.values():
        for plfiles in pfiles.values():
            random.shuffle(plfiles)

            part_div = len(plfiles) // 10
            part_mod = len(plfiles) % 10
            for i in range(10):
                partitions[i] = partitions[i] + [filename for filename in plfiles[i*part_div:(i+1)*part_div]]
            
            partitions = sorted(partitions, key= lambda x: len(x))
            for i,f in enumerate(range(10*part_div,10*part_div+part_mod)):
                partitions[i].append(plfiles[f])
    return partitions
    
    
    
def one_out_kfold(data):
    """Split the movements into 8 partitions of approximately equal size according to the "all-in" strategy. The movements of the i-th partition are selected as the movements of the i-th participant.

    Args:
        data (dictionary): A dictionary with the movement filename as the key and a pandas dataframe containing the corresponding skeletal data of the movement as the value.

    Returns:
        partitions (list): A list containing 8 lists, one for each partition. The nested lists contain the movement filenames of the partition.
    """

    part_movements = {f"P{p}":[] for p in range(1,9)}
    for mov in data.keys():
        part_movements[mov[:2]].append(mov)
    partitions = part_movements.values()
    return partitions