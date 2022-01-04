import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os.path import isfile
from ..config import data_dir, results_dir, fit_dir, fit1_dir
import pandas as pd
from ..lib.preprocessing_data.preprocessing import generate_epoch,butter_bandpass_filter,generate_combine_data,apply_pyriemann_data
from ..lib.tools.utils import check_create_dir
from ..lib.tools.visualization import my_plot_roc_curve, visualize_confusion_matrix, visualize_roc_curve, visualize_classification_report

channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
    'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
    'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 
    'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']



#train_labels = pd.read_csv(data_dir+"TrainLabels.csv")

y_train = pd.read_csv(data_dir+'TrainLabels.csv')['Prediction'].values
y_test = pd.read_csv(data_dir+'true_labels.csv')['Prediction'].values


if not isfile(fit_dir+'train_data.npy') or not isfile(fit_dir+'test_data.npy'):
    print('Prepare to generate combine data !')
    generate_combine_data(fs=200,channels=channels,filter=butter_bandpass_filter)
    # load data
    train_data = np.load(fit_dir+'train_data.npy', allow_pickle=True)
    test_data  = np.load(fit_dir+'test_data.npy', allow_pickle=True)
if not isfile(fit1_dir+'X_train.npy') or not isfile(fit1_dir+'X_test.npy'):
    print('Prepare data for train and test !')
    apply_pyriemann_data(train_data, test_data)





X_train = np.load(fit1_dir+'X_train.npy', allow_pickle=True)
X_test = np.load(fit1_dir+'X_test.npy', allow_pickle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

# predict
y_pred_val = clf.predict(X_val)
y_pred_test=clf.predict(X_test)
print('Accuracy  for validation set = ', accuracy_score(y_pred_val,y_val))
print('Accuracy  for test set = ', accuracy_score(y_pred_test,y_test))

clf_report = classification_report(y_pred_test, y_test,output_dict=True)
print('Report',clf_report)
visualize_classification_report(clf_report,model_name='RandomForest')


# Save prediction 
test_labels = pd.read_csv(data_dir+"SampleSubmission.csv")
test_labels['Prediction']=y_pred_test
print(test_labels.head())
check_create_dir(results_dir)
test_labels.to_csv(results_dir+"Submission_RandomForest.csv",index=None)



my_plot_roc_curve(clf,X_test,y_test,model_name='RandomForest', fig_name='RandomForest_test')
visualize_confusion_matrix(clf,X_test,y_test,title='RandomForest',fig_name='RandomForest_test',normalize='true')
#visualize_roc_curve([clf],X_test,y_test,title='RandomForest',fig_name='RandomForest_test')