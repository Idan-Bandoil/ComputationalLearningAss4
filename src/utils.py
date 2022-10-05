import csv
import scipy
import os.path
import scipy.io.arff
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from functools import partial
from sklearn.naive_bayes import GaussianNB
from sklearn import feature_selection as fs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF as RELIEFF
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from Algorithms.MFMW import mfmw
from Algorithms.MFMW_New import mfmw_new
from Algorithms.Tri_Stage import tri_stage


corpus_1_mat = ['ORL', 'Carcinom', 'colon', 'nci9', 'pixraw10P']
corpus_2_arff = ['CNS', 'Leukemia_3c', 'Lung', 'MLL', 'Ovarian']
corpus_3_bioconductor = ['CLL', 'DLBCL', 'bcellViper', 'leukemiasEset', 'bladderbatch']
corpus_4_microbiomicdata = ['FS', 'CBH', 'CS', 'CSS', 'FSH']


dataset_fields = ['DatasetName', 'NumOfSamples', 'OriginalNumOfFeatures']
experiment_fields = ['FilteringAlgorithm', 'LearningAlgorithm', 'NumFeaturesSelected(K)', 'CVMethod', 'Folds']
prediction_fields = ['MeasureType', 'MeasureVal', 'ListOfFeaturesNames', 'ListOfFeaturesScores', 'FeatureSelectionTime', 'FitTime', 'InferenceTime']
fields = dataset_fields + experiment_fields + prediction_fields


def auc_scorer(y_test, y_pred, **kwargs):
    classes = kwargs['classes']
    test_classes = np.unique(y_test)
    if len(test_classes) == 1: # Only one class in the test set
        return accuracy_score(y_test, y_pred)
    elif len(test_classes) == 2: # Binary classification in the test set
        return roc_auc_score(y_test, y_pred)
    else: # Multi-class
        y_pred_proba = np.array([[1 if y_pred[i] == classes[j] else 0 for j in range(len(classes))] for i in range(len(y_pred))])
        return roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=classes)
        
def prauc_scorer(y_test, y_pred, **kwargs):
    classes = kwargs['classes']
    test_classes = np.unique(y_test)
    if len(test_classes) == 1: # Only one class in the test set
        return average_precision_score(y_test, y_pred)
    elif len(test_classes) == 2: # Binary classification in the test set
        return average_precision_score(y_test, y_pred)
    else: # Multi-class
        y_pred_proba = np.array([[1 if y_pred[i] == classes[j] else 0 for j in range(len(classes))] for i in range(len(y_pred))])
        return sum(average_precision_score([1 if label == test_classes[i] else 0 for label in y_test], y_pred_proba[:, i]) for i in range(len(test_classes))) / len(test_classes)


def make_loader(dataset_name):
    if dataset_name in corpus_1_mat:
        def load_mat():
            path = f'../data/mat/{dataset_name}.mat'
            mat = scipy.io.loadmat(path)
            return mat['X'], mat['Y'].ravel()
        return load_mat
    elif dataset_name in corpus_2_arff:
        def load_arff():
            path = f'../data/arff/{dataset_name}.arff'
            arff = scipy.io.arff.loadarff(path)
            df = pd.DataFrame(arff[0])
            return df.iloc[:, :-1].values, df.iloc[:, -1].values
        return load_arff
    elif dataset_name in corpus_3_bioconductor:
        def load_csv():
            path = f'../data/bioconductor/{dataset_name}.csv'
            df = pd.read_csv(path, index_col=[0])
            df = df.T
            return df.iloc[:, 1:].values, df.iloc[:, 0].values
        return load_csv
    elif dataset_name in corpus_4_microbiomicdata:
        def load_csv():
            path = f'../data/microbiomicdata/{dataset_name}.csv'
            df = pd.read_csv(path)
            return df.iloc[:, :-1].values, df.iloc[:, -1].values
        return load_csv


def describe_dataset(name, X, y):
    return {
        'DatasetName': name,
        'NumOfSamples': X.shape[0],
        'OriginalNumOfFeatures': X.shape[1],
    }
    

def is_preprocessed(dataset_name):
    return os.path.isfile(f'../preprocesseddata/{dataset_name}_processed.csv')


def save_preprocessed(dataset_name, X, y):
    df = pd.DataFrame(X)
    df['class'] = y
    df.to_csv(f'../preprocesseddata/{dataset_name}_processed.csv', index=False)


def load_preprocessed(dataset_name):
    df = pd.read_csv(f'../preprocesseddata/{dataset_name}_processed.csv')
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


# mRMR
def mrmr(X, y):
    return np.array(MRMR.mrmr(X, y))


# SelectFDR with FDR = 0.1 and f_classif evaluation
def select_fdr(X, y):
    selector = fs.SelectFdr(alpha=0.1, score_func=fs.f_classif)
    selector.fit(X, y)
    return selector.scores_


# RFE with SVM evaluation
def rfe_svm(X, y, k=None):
    rfe = fs.RFE(SVC(kernel='linear'), n_features_to_select=k)
    rfe.fit(X, y)
    return rfe.ranking_


# reliefF method
def reliefF(X, y):
    return RELIEFF.reliefF(X, y)


def get_fs(fsMethod, k=None):
    if fsMethod == 'MRMR':
        return mrmr
    if fsMethod == 'FDR':
        return select_fdr
    if fsMethod == 'RFE_SVM':
        return partial(rfe_svm, k=k)
    if fsMethod == 'ReliefF':
        return reliefF
    if fsMethod == 'MFMW':
        return partial(mfmw, k=k)
    if fsMethod == 'TRI_STAGE':
        return partial(tri_stage, k=k)
    if fsMethod == 'MFMW_New':
        return partial(mfmw_new, k=k)
    

def get_classifier(method):
    if method == 'KNN':
        return KNeighborsClassifier(n_jobs=-1)
    if method == 'SVM':
        return SVC()
    if method == 'NB':
        return GaussianNB()
    if method == 'RF':
        return RandomForestClassifier(n_jobs=-1)
    if method == 'LogReg':
        return LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1)
    

def get_cv_method(n_samples, y=None):
    if n_samples <= 50:
        return ('Leave-pair-out', LeavePOut(2))
    elif n_samples <= 100:
        return ('Leave-one-out', LeaveOneOut())
    elif n_samples < 1000:
        if y is not None and np.min(np.bincount(y)) > 10:
            return ('10Fold', StratifiedKFold(n_splits=10, shuffle=True, random_state=10))
        return ('10Fold', KFold(n_splits=10, shuffle=True, random_state=10))
    else:
        if y is not None and np.min(np.bincount(y)) > 5:
            return ('5Fold', StratifiedKFold(n_splits=5, shuffle=True, random_state=10))
        return ('5Fold', KFold(n_splits=5, shuffle=True, random_state=10))


def delete_data():
    if os.path.exists('data.csv'):
        os.remove('data.csv')


def row_exists(row_to_search):
    if not os.path.isfile('data.csv'):
        return False
        
    # turn all values in the row dict to string
    row_to_search = {k: str(v) for k, v in row_to_search.items()}
    with open('data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if all(item in row.items() for item in row_to_search.items()):
                return True
    return False

rows = []

def append_row(row_to_insert=None, finish=False):
    global rows
    if finish:
        for row in rows:
            with open('data.csv', 'a') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writerow(row)
        rows = []
        return

    if len(rows) < 5000:
        rows.append(row_to_insert)
        return
    rows.append(row_to_insert)
    
    if not os.path.isfile('data.csv'):
        with open('data.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            
    for row in rows:
        with open('data.csv', 'a') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writerow(row)
    rows = []
