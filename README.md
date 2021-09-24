# Human action prediction from hand movement for human-robot collaboration
## Dependencies
- Python version: >= 3.3

This repository has also the following package dependencies:
- numpy
- pandas
- scipy
- seaborn
- matplotlib
- scikit_learn

## Things to do before running the script
- Clone the repository to your local machine
- Unzip the data.tar.gz file in the dataset folder.
```
tar -xf data.tar.gz
```
- Install the repository's dependencies.


## How to run the evaluation script ?
In order to print the help menu of the evaluation script, run the following command:
```python
python3 evaluate_ml.py --help #or
python3 evaluate_ml.py -h
```

The help menu that is printed is the following:
```
usage: evaluate_ml.py [--feature_sets {1,2,3,4,5,6,7,8} [{1,2,3,4,5,6,7,8} ...]] [--methods {RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} [{RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} ...]] [--plot]
                      [-h]
                      {all-in,one-out}

positional arguments:
  {all-in,one-out}      The dataset split strategy to be used.

optional arguments:
  --feature_sets {1,2,3,4,5,6,7,8} [{1,2,3,4,5,6,7,8} ...]
                        The feature sets that will be used in order to produce the kinematic features of the grasping movement.
  --methods {RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} [{RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} ...]
                        The methods that will be used in order to train and evaluate the model.
  --plot                Plot the accuracries of the different methods superimposed and the confusion matrix of each method.
  -h, --help            Show this help message and exit.
```

### Dataset Split Strategy
As specificied in the help menu above, the user has to select the dataset split strategy to be used. The dataset split strategies are the following:
- all-in
- one-out

```python
python3 evaluate_ml.py all-in   #Run the evaluation script for the 'all-in' dataset split strategy.
python3 evaluate_ml.py one-out  #Run the evaluation script for the 'one-out' dataset split strategy.
```

### Methods
In the previous cases, the script will output the results of all the methods:
- RandomForest
- GradientBoosting
- ExtraTrees
- SVM
- GaussianProcess

for each of the 8 feature sets.

### Feature Sets
The kinematic features that are engineered for each feature set are the following:
<p align="center">
<img src="assets/feature_sets.png" height=600></img>
</p>

It is possible to specify the methods and the feature sets for which the models will be evaluated.

e.g.
```python
python3 evaluate_ml.py one-out --feature_sets 5 3 1
python3 evaluate_ml.py one-out --methods ExtraTrees RandomForest SVM 
python3 evaluate_ml.py all-in --feature_sets 5 3 1 --methods ExtraTrees RandomForest SVM
```
### Results Format
The results are printed in the following format:
```
python3 evaluate_ml.py all-in --feature_sets 5 3 1 --methods ExtraTrees RandomForest SVM
```

```
Dataset split strategy: 'all-in'
feature set: 1 ['index-middle aperture', 'thumb-index aperture', 'thumb-middle aperture']
               20    40    60    80    100
SVM          68.26 68.97 72.73 76.07 82.64
RandomForest 76.21 80.02 80.99 86.15 89.79
ExtraTrees   75.38 82.39 82.39 87.56 90.91

Dataset split strategy: 'all-in'
feature set: 3 ['index-middle aperture', 'thumb-index aperture', 'thumb-middle aperture', 'wrist x-std_dev', 'wrist y-std_dev']
               20    40    60    80    100
SVM          69.64 71.04 72.87 75.39 81.81
RandomForest 79.72 80.13 82.24 86.29 90.21
ExtraTrees   78.03 82.36 83.90 88.24 91.04

Dataset split strategy: 'all-in'
feature set: 5 ['index-middle aperture', 'thumb-index aperture', 'thumb-middle aperture', 'wrist x-speed', 'wrist y-speed']
               20    40    60    80    100
SVM          36.77 42.12 48.82 53.41 55.27
RandomForest 77.05 78.72 81.94 84.75 88.66
ExtraTrees   77.18 81.66 82.50 86.28 90.49
```

For each method, movement completion percentage and feature set combination, the floating point number indicates the average accuracy of the model in the k-fold evaluation scheme.

<b>Note:</b> The models' accuracy results will NOT be the same for two executions of the evaluation script with the same arguemts. 

### Plotting the results
It is also possible to plot the accuracy results and the confusion matrices of the models interactively.

Each confusion matrix has a 3x3 shape and was calculated as the average of k confusion matrices in the k-fold evaluation scheme. The floating point number outside the parentheses indicates the average accuracy for a (ground-truth label, predicted class) combination. On the other hand, the floating point number inside the parentheses indicates the standard deviation of the k accuracies for a (ground-truth label, predicted class) combination.


e.g.
```
python3 evaluate_ml.py all-in --feature_sets 5 3 1 --methods ExtraTrees RandomForest SVM --plot
```
The user can choose the feature set and the method for which the plots are drawn interactively, using the arrow keys:
- the 'up' and 'down' arrow keys are used to change the feature set
- the 'right' and 'left' arrow keys are used to change the method

e.g. 
```
python3 evaluate_ml.py all-in --feature_sets 5 3 1 --methods ExtraTrees RandomForest SVM
```
Initally the following figures are plotted:

<p align="center">
<img src="assets/acc_ex1.png"></img>
</p>

<p align="center">
<img src="assets/confmtx_et_ex1.png"></img>
</p>

After pressing the 'right' key, the confusion matrices' figure is updated:

<p align="center">
<img src="assets/confmtx_rf_ex1.png"></img>
</p>

After pressing the 'up' key, both the accuracy and the confusion matrices figures are updated:

<p align="center">
<img src="assets/acc_ex3.png"></img>
</p>

<p align="center">
<img src="assets/confmtx_rf_ex3.png"></img>
</p>