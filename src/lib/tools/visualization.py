import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from ...config import results_dir
from .utils import check_create_dir


def my_plot_roc_curve(clf,X_test,y_test, model_name,fig_name=None):
    y_pred_proba = clf.predict_proba(X_test)
    r_roc_auc = roc_auc_score(y_test, y_pred_proba[:,1])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    if fig_name == None:
        fig_name = model_name
    path_fig = results_dir+ fig_name + '_ROC.png'

    plt.figure()
    plt.plot(fpr, tpr, label= model_name+' (area = %0.2f)' % r_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    check_create_dir(results_dir)
    plt.savefig(path_fig)
    plt.show()

def visualize_confusion_matrix(clf,X_test,y_test,title,fig_name=None,normalize='true'):
    if fig_name == None:
        fig_name = title

    
    disp = plot_confusion_matrix(clf, X_test, y_test,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    check_create_dir(results_dir)
    plt.savefig(results_dir+ fig_name + '_confusion_matrix.png')
    plt.show()

    print(title)
    print(disp.confusion_matrix)
def visualize_roc_curve(list_clf,X_test,y_test, title,fig_name=None):
    
    if fig_name == None:
        fig_name = title
    ax = plt.gca()

    
    for i in  range(0,len(list_clf)):
        clf_disp = plot_roc_curve(list_clf[i], X_test, y_test,ax=ax, alpha=0.8)
        
    clf_disp.ax_.set_title(title)
    plt.plot([0, 1], [0, 1],'r--')
    check_create_dir(results_dir)
    plt.savefig(results_dir+ fig_name + '_ROC_v2.png')
    plt.show()
def visualize_classification_report(clf_report,model_name):
    # .iloc[:-1, :] to exclude support
    plt.figure(figsize = (10,5))
    sns_plot= sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(),rotation=0)
    sns_plot.figure.savefig(results_dir+model_name+'_clf_report.png')
