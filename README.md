# Object size prediction from hand movement using a single RGB sensor
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
usage: evaluate_ml.py [--feature_sets {1,2,3,4,5,6,7,8} [{1,2,3,4,5,6,7,8} ...]]
                      [--methods {RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} [{RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} ...]] [--plot] [--seed SEED] [-h]
                      {all-in,one-out}

positional arguments:
  {all-in,one-out}      The dataset split strategy to be used.

optional arguments:
  --feature_sets {1,2,3,4,5,6,7,8} [{1,2,3,4,5,6,7,8} ...]
                        The feature sets that will be used in order to produce the kinematic features of the grasping movement.
  --methods {RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} [{RandomForest,GradientBoosting,ExtraTrees,SVM,GaussianProcess} ...]
                        The methods that will be used in order to train and evaluate the model.
  --plot                Plot the accuracries of the different methods superimposed and the confusion matrix of each method.
  --seed SEED           A seed to initialize the random generator.
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
                     20          40          60          80          100
ExtraTrees    77.1 (5.0)  82.4 (4.2)  82.8 (4.1)  88.4 (3.5)  92.0 (3.3)
RandomForest  77.0 (4.8)  79.4 (4.2)  82.1 (3.9)  87.0 (3.0)  89.4 (3.2)
SVM           67.4 (2.3)  69.0 (3.8)  73.4 (2.5)  75.7 (3.7)  82.7 (3.0)

Dataset split strategy: 'all-in'
feature set: 3 ['index-middle aperture', 'thumb-index aperture', 'thumb-middle aperture', 'wrist x-std_dev', 'wrist y-std_dev']
                     20          40          60          80          100
ExtraTrees    78.5 (2.9)  82.1 (4.3)  84.6 (3.4)  88.1 (2.4)  91.0 (2.3)
RandomForest  78.9 (4.3)  80.1 (3.7)  81.4 (2.8)  86.4 (3.1)  88.5 (3.2)
SVM           69.5 (5.2)  70.8 (2.9)  71.9 (4.8)  75.1 (5.1)  82.1 (4.5)

Dataset split strategy: 'all-in'
feature set: 5 ['index-middle aperture', 'thumb-index aperture', 'thumb-middle aperture', 'wrist x-speed', 'wrist y-speed']
                     20          40          60          80          100
ExtraTrees    77.3 (5.0)  80.8 (4.3)  83.6 (3.1)  87.4 (3.0)  90.8 (4.3)
RandomForest  75.9 (6.1)  80.0 (2.0)  82.0 (3.6)  85.4 (5.1)  87.3 (4.4)
SVM           36.2 (6.1)  42.9 (5.4)  49.0 (6.3)  53.3 (5.2)  56.2 (3.9)
```

For each method, movement completion percentage and feature set combination, the floating point numbers indicate the mean and the standard deviation of the model's accuracy rates in the k-fold evaluation scheme.

<b>Note:</b> The models' accuracy results will NOT be the same for two executions of the evaluation script with the same arguments unless a seed is provided. 

### Plotting the results
It is also possible to plot the accuracy results and the confusion matrices of the models interactively.

Each confusion matrix has a 3x3 shape and was calculated as the average of k confusion matrices in the k-fold evaluation scheme. The floating point number outside the parentheses indicates the average accuracy for a (ground-truth label, predicted class) combination. On the other hand, the floating point number inside the parentheses indicates the standard deviation of the k accuracies for a (ground-truth label, predicted class) combination.

The user can choose the feature set and the method for which the plots are drawn interactively, using the arrow keys:
- the 'up' and 'down' arrow keys are used to change the feature set
- the 'right' and 'left' arrow keys are used to change the method

e.g.
```
python3 evaluate_ml.py all-in --feature_sets 5 3 1 --methods ExtraTrees RandomForest SVM --plot
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

The program terminates when the user closes one of the figures.

## How to reproduce the results of the paper?
Seed the random generator with 0.
```
python3 evaluate_ml.py all-in --seed 0
python3 evaluate_ml.py one-out --seed 0
```