# Functional Brain Imaging
Author: Le Tran Ngoc Tran

Project from the MVA course FBI.

## Objective 
The challenge is to predict, based on the user’s response to the feedback event, whether feedback was positive or negative.

## Requirements
Python 3.6
Scikit-learn 0.22
PyRiemann 0.2.6
Seaborn
Tensorflow



## Run the  submission
Just run this command in a terminal:

```bash
python start.py
```


### Experiments:
 

SVM
```bash
python -m src.experiments.exp_svm

```
LogisticRegression
```bash
python -m src.experiments.exp_LogisticRegression
```
RandomForest
```bash
python -m src.experiments.exp_RandomForest
```
Neural Networks
```bash
python -m src.experiments.exp_DeepLearning
```

## Resources

| Path | Description
| :--- | :----------
| [Kernel]() | Main folder.
| &boxvr;&nbsp; [data]() | data folder.
| &boxvr;&nbsp; [fit]() | Folder to store the processing data.
| &boxvr;&nbsp; [results]() | Store the results of the classification and some figures.
| &boxvr;&nbsp; [src]() | the main source codes.
| &boxv;&nbsp; &boxvr;&nbsp; [lib]() | library.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [preprocessing_data]() | functions for preprocessing data.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [tools]()   | utils and visualization.
| &boxv;&nbsp; &boxvr;&nbsp; [experiments]() | some experiments test cases.
| &boxv;&nbsp; &boxvr;&nbsp; [config.py]() | some configurations.
| &boxvr;&nbsp; [start.py]() | run the test case.