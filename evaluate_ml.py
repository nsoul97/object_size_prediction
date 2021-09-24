import sklearn.ensemble, sklearn.svm, sklearn.gaussian_process, sklearn.metrics
import argparse
import numpy as np
import pandas as pd
from preprocess import read_dataset, preprocess_dataset
from ml_features import feature_engineering, feature_statistics_extraction
from setup_data import setup_train_test_split, setup_fvecs_labels
from plot_results import plot_results

def parse_args():
    """ Create the help menu prompts and parse the given attributes.
        If an optional argument is not provided in the command line, the default attribute values are used. 
        If the user provides the same valid feature set id or the same valid method multiple times, the duplicates are discarded.

    Returns:
        strategy (string): The dataset split strategy string ('all-in', 'one-out')
        feature_sets (set): A set of integers containing the feature sets ids ([1,8])
        methods (set): A set of strings containing the names of the learning algorithms to be used ('RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess')
        plot (boolean): A boolean that determines wheter the evaluation results are plotted.

    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("strategy", type=str, choices=['all-in', 'one-out'],
                        help="The dataset split strategy to be used.")

    parser.add_argument("--feature_sets", nargs='+', type=int, default=list(range(1,9)), choices=list(range(1,9)),
                        help="The feature sets that will be used in order to produce the kinematic features of the grasping movement.")

    parser.add_argument("--methods", nargs='+', type=str, default=['RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess'], 
                        choices=['RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess'],
                        help="The methods that will be used in order to train and evaluate the model.")

    parser.add_argument("--plot", action='store_true',
                        help='Plot the accuracries of the different methods superimposed and the confusion matrix of each method.')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()

    strategy = args.strategy
    feature_sets = set(args.feature_sets)
    methods = set(args.methods)
    plot = args.plot

    return strategy, feature_sets, methods, plot


def fsid_to_fsnames(fsid):
    """ Returns the names of the kinematic features that correspond to the given feature set id. The empty string is also included in the names, because the 'points number' summary statistic is computed once for all the 
        the kinematic features.

    Args:
        fsid (int): The feature set id is an integer in [1,8].

    Returns:
        fsnames (set): A set containing the names of the kinematic features that belong in the given feature set.
    """

    fs = {  1: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture'},
            2: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist x-coord', 'wrist y-coord'},
            3: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist x-std_dev', 'wrist y-std_dev'},
            4: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist xy-std_dist'},
            5: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist x-speed', 'wrist y-speed'},
            6: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist xy-speed'},
            7: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist x-coord', 'wrist y-coord', 'wrist x-std_dev', 'wrist y-std_dev'},
            8: {'', 'thumb-index aperture', 'thumb-middle aperture', 'index-middle aperture', 'wrist x-coord', 'wrist y-coord', 'wrist x-speed', 'wrist y-speed'}
        }
    
    fsnames = fs[fsid]
    return fsnames

def train(train_set, method):
    """ Train a model based on the given training set and return it. The model to be trained is specified via the method string.

    Args:
        train_set (tuple): The training set contains the feature vectors of the training grasping movements and their corresponding ground truth labels (S:0, M:1, L:2) 
        method (string): A string that specifies the model to be trained ('RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM' or 'GaussianProcess')

    Returns:
        model (RandomForestClassifier || GradientBoostingClassifier || ExtraTreesClassifier || SVC || GaussianProcessClassifier): The model that was trained based on the training set's instances.
    """

    train_x, train_y = train_set

    if method == 'RandomForest':
        model = sklearn.ensemble.RandomForestClassifier()
        model.fit(train_x, train_y)
    elif method == 'GradientBoosting':
        model = sklearn.ensemble.GradientBoostingClassifier()
        model.fit(train_x, train_y)
    elif method == 'ExtraTrees':
        model = sklearn.ensemble.ExtraTreesClassifier()
        model.fit(train_x, train_y)
    elif method == 'SVM':
        model = sklearn.svm.SVC()
        model.fit(train_x, train_y)
    elif method == 'GaussianProcess':
        model = sklearn.gaussian_process.GaussianProcessClassifier()
        model.fit(train_x, train_y)
    else:
        assert(0)

    return model

def predict(test_set, model):
    """ The model predicts the object-to-be-grasped for each of the test grasping movements. The predictions are compared to the ground truth labels in order to return the model's accuracy and confusion matrix on the test
        set.


    Args:
        test_set (tuple): The test set contains the feature vectors of the test grasping movements and their corresponding ground truth labels (S:0, M:1, L:2) 
        model (RandomForestClassifier || GradientBoostingClassifier || ExtraTreesClassifier || SVC || GaussianProcessClassifier): The model that was trained based on the training set's instances.

    Returns:
        acc (np.float64): The accuracy of the model on the test set.
        conf_mtx (np.ndarray): The 3x3 confusion matrix of the model on the test set.
    """

    test_x, test_y = test_set
    pred_y = model.predict(test_x)

    acc = sklearn.metrics.accuracy_score(test_y, pred_y, normalize=True)
    conf_mtx = sklearn.metrics.confusion_matrix(test_y, pred_y, normalize='true')

    return acc, conf_mtx

def print_results(acc_dict, strategy, fs_ids, methods):
    """ Print the accuracy results for the given dataset split strategy, feature sets and methods for the movement completion percentages 20%, 40%, 60%, 80%, 100%.

    Args:
        acc_dict (dictionary): A 3-level dictionary with a feature set id as the 1st-level key, a movement completion percentage as the 2nd-level key and a method as the 3rd-level key. The value of the dictionary in the
                               3rd level is a tuple, whose 1st value indicates the accuracy of the method for the given feature set and movement completion percentage. This accuracy was calculated as
                               the average of k accuracy results in the k-fold evaluation scheme. The 2nd value of the tuple is the standard deviation of the k accuracy results. 
        strategy (string): The dataset split strategy string ('all-in', 'one-out')
        fs_ids (list): A list of integers containing the feature sets ids ([1,8])
        methods (list): A list of strings containing the names of the learning algorithms to be used ('RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess')
    """

    MOV_COMPLETION = [20, 40, 60, 80, 100]
    pd.options.display.float_format = "{:,.2f}".format

    for fs_id in sorted(fs_ids):
        fs_names = sorted(fsid_to_fsnames(fs_id))
        fs_names.remove('')

        print(f"Dataset split strategy: '{strategy}'")
        print(f"feature set: {fs_id} {fs_names}")

        results_df = pd.DataFrame(data={mov_compl:[acc_dict[fs_id][mov_compl][method][0]*100 for method in methods] for mov_compl in MOV_COMPLETION},
                                  index=methods)

        print(results_df, end="\n\n")
    
    

def main():
    """ 1) Parse the command line arguments.
        2) Read the dataset and apply the preprocessing pipeline to identify and isolate the grasping phase of the movements.
        3) Engineer the kinematic features for each of the grasping movements and extract the feature statistics for each of the 20%, 40%, 60%, 80% and 100% movement completion percentages.
        4) Train and evaluate each model for the given dataset split strategy, for each of the 20%, 40%, 60%, 80% and 100% movement completion intervals and for each of the given feature sets.
        5) Print the accuracy of the models for the given dataset split strategy, for each of the 20%, 40%, 60%, 80% and 100% movement completion intervals and for each of the given feature sets.
        6) [Optional] Plot the accuracies and the confusion matrices interactively. 
    """
    MOV_COMPLETION = [20, 40, 60, 80, 100]

    strategy, fs_ids, methods, plot = parse_args()

    data = read_dataset(joints=["RWrist", "RThumb4FingerTip", "RIndex4FingerTip", "RMiddle4FingerTip"]) 
    preprocess_dataset(data)
    feature_engineering(data)
    partial_data = feature_statistics_extraction(data)
    
    acc_dict = {fs_id:{mov_completion_perc:{method:[] for method in methods} for mov_completion_perc in MOV_COMPLETION} for fs_id in fs_ids}
    conf_mtx_dict = {fs_id:{mov_completion_perc:{method:[] for method in methods} for mov_completion_perc in MOV_COMPLETION} for fs_id in fs_ids}

    for fs_id in fs_ids:
        fs_names = fsid_to_fsnames(fs_id)
        for train_set_names, test_set_names in setup_train_test_split(data, strategy):
            for mov_completion_perc in MOV_COMPLETION:

                train_set = setup_fvecs_labels(train_set_names, mov_completion_perc, partial_data, fs_names)
                test_set = setup_fvecs_labels(test_set_names, mov_completion_perc, partial_data, fs_names)

                for method in methods:
                    model = train(train_set, method)
                    acc, conf_mtx = predict(test_set, model)
                    
                    acc_dict[fs_id][mov_completion_perc][method].append(acc)
                    conf_mtx_dict[fs_id][mov_completion_perc][method].append(conf_mtx)
    
    for fs_id in fs_ids:
        for mov_completion_perc in MOV_COMPLETION:
            for method in methods:
                acc_dict[fs_id][mov_completion_perc][method] = (np.average(acc_dict[fs_id][mov_completion_perc][method]), np.std(acc_dict[fs_id][mov_completion_perc][method]))
                conf_mtx_dict[fs_id][mov_completion_perc][method] = (np.average(conf_mtx_dict[fs_id][mov_completion_perc][method], axis=0), np.std(conf_mtx_dict[fs_id][mov_completion_perc][method], axis=0))

    print_results(acc_dict, strategy, fs_ids, methods)
    if plot: plot_results(acc_dict, conf_mtx_dict, strategy, fs_ids, methods)

if __name__ == '__main__':
    main()
