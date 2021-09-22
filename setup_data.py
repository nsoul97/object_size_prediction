import random
import numpy as np


def all_in_kfold(data):
    """ Split the movements into 10 partitions of approximately equal size (+-1) according to the "all-in" strategy. The movements of each partition are selected randomly.
        The partitions have approximately the same number of movements (+-1) for a given (participant, object) pair.

    Args:
        data (dictionary): A dictionary with the movement filename as the key and a pd.DataFrame containing the corresponding data of the movement as the value. This dictionary contains all the movements
                           of the dataset.

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
    """ Split the movements into 8 partitions of approximately equal size according to the "all-in" strategy. The movements of the i-th partition are selected as the movements of the i-th participant.

    Args:
        data (dictionary): A dictionary with the movement filename as the key and a pd.DataFrame containing the corresponding data of the movement as the value. This dictionary contains all the movements
                           of the dataset.

    Returns:
        partitions (list): A list containing 8 lists, one for each partition. The nested lists contain the movement filenames of the partition.
    """

    part_movements = {f"P{p}":[] for p in range(1,9)}
    for mov in data.keys():
        part_movements[mov[:2]].append(mov)
    partitions = list(part_movements.values())
    return partitions

def setup_train_test_split(data, strategy):
    """ Yield one of the k training-test set splits based on the k input partitions. When this generator is called for the i-th time, the i-th partition is selected as test set and the other partitions
        are concatenated to form the the training set.

    Args:
        data (dictionary): A dictionary with the movement filename as the key and a pd.DataFrame containing the corresponding data of the movement as the value. This dictionary contains all the movements
                           of the dataset.
        strategy (str): A string that determines the dataset split strategy to be used. The strategy string acceptable values are 'all-in' and 'one-out'.

    Yields:
        split (tuple): A tuple of 2 lists. The first list contains the filenames of the training set and the second list contains the filenames of the testing set.
    """
    if strategy == 'all-in':
        partitions = all_in_kfold(data)
    elif strategy == 'one-out':
        partitions = one_out_kfold(data)
    else:
        assert(0)
    
    for i in range(len(partitions)):
        train_set = list(np.concatenate([partitions[j] for j in range(len(partitions)) if i!=j]))
        test_set = partitions[i]
        split = (train_set, test_set)
        yield split

def setup_fvecs_labels(mov_names, mov_completion_perc, partial_data, fs_names):
    """ Return the feature vectors and the ground truth labels that are associated with the given movement names and the given movement completion percentage.
        The feature vectors include only the summary statistics that were extracted for the given kinematic features.
        The ground truth labels 'S', 'M', 'L' are replaced by the 0, 1, 2 corresponding numerical values.

    Args:
        mov_names (list): A list containing the movement filenames.
        mov_completion_perc (int): The movement completion percentage for which the feature vector of a movement is calculated.
        partial_data (dictionary): A dictionary with a "<movement filename>_<movement completion percentage>" string as key and a pd.Series containing the summary statistics of the kinematic features for
                                   the corresponding movement and movement completion percentage.
        fs_names (set): A set containing the names of the kinematic features that belong in the given feature set.

    Returns:
        dataset_fvecs (list): A list containg the feature vectors of the given movements. Each feature vector is represented as a numpy array with shape (1+7*F,), where F is the number of the selected kinematic features.
        dataset_labels (list): A list containing the numerical values of the ground-truth labels of the given movements.
    """
    labels_dict = {'S':0, 'M':1, 'L':2}

    dataset_fvecs = [partial_data[f"{mov_name}_{mov_completion_perc}"][fs_names].to_numpy() for mov_name in mov_names]
    dataset_labels = [labels_dict[mov_name.split('_')[1]] for mov_name in mov_names]

    return dataset_fvecs, dataset_labels