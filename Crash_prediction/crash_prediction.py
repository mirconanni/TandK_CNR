import os
import sys
sys.path.append('..')
import json
import pickle
import numpy as np
import pandas as pd
import datetime
from os import listdir
import warnings

warnings.filterwarnings("ignore")  #, category=UndefinedMetricWarning)

import sklearn; print("Scikit-Learn", sklearn.__version__)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import MinMaxScaler
import imblearn; print("IMB-Learn", imblearn.__version__)
from imblearn.over_sampling import SMOTE  # , RandomOverSampler, ADASYN

import Crash_prediction.crash_config as cfg

__author__ = 'Riccardo Guidotti'

params = {
    'DT': {
        'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'class_weight': [None, 'balanced',
                         {0: 0.1, 1: 0.9}, {0: 0.2, 1: 0.8}, {0: 0.3, 1: 0.7}, {0: 0.4, 1: 0.6},
                         {0: 0.6, 1: 0.4}, {0: 0.7, 1: 0.3}, {0: 0.8, 1: 0.2}, {0: 0.9, 1: 0.1}, ],
        'random_state': [0],
    },
    'RF': {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'bootstrap': [False, True],
        'class_weight': [None, 'balanced',
                         {0: 0.1, 1: 0.9}, {0: 0.2, 1: 0.8}, {0: 0.3, 1: 0.7}, {0: 0.4, 1: 0.6},
                         {0: 0.6, 1: 0.4}, {0: 0.7, 1: 0.3}, {0: 0.8, 1: 0.2}, {0: 0.9, 1: 0.1}, ],
        'random_state': [0],
    },
    'AB': {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0],
        'random_state': [0],
    },
    'ET': {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'bootstrap': [False, True],
        'class_weight': [None, 'balanced',
                         {0: 0.1, 1: 0.9}, {0: 0.2, 1: 0.8}, {0: 0.3, 1: 0.7}, {0: 0.4, 1: 0.6},
                         {0: 0.6, 1: 0.4}, {0: 0.7, 1: 0.3}, {0: 0.8, 1: 0.2}, {0: 0.9, 1: 0.1}, ],
        'random_state': [0],
    },
}

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def train_test_classifier(df_train, df_test, features, class_name, clf_type, scoring, oversampling,
                          path, area, index, feature_type, random_state):

    # sm = SMOTE(random_state=random_state, k_neighbors) #, sampling_strategy='not majority')
    # ros = RandomOverSampler(random_state=random_state)
    # sm = ADASYN(random_state=random_state)
    mms = MinMaxScaler()

    X_train = df_train[features].values
    Y_train = df_train[class_name].values
    X_train = mms.fit_transform(X_train)
    # print(np.any(np.isinf(X_train)), np.any(np.isnan(X_train)), np.all(np.isfinite(X_train)))
    # print(np.any(np.isinf(Y_train)), np.any(np.isnan(Y_train)), np.all(np.isfinite(Y_train)))
    aaa = np.unique(Y_train, return_counts=True)
    k_neighbors = min(5, np.unique(Y_train, return_counts=True)[1][1] - 1)
    # k_neighbors = min(5, np.unique(Y_train, return_counts=True)[1] - 1)
    if k_neighbors > 1:
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_bal, Y_train_bal = sm.fit_resample(X_train, Y_train)
    else:
        X_train_bal, Y_train_bal = X_train, Y_train

    X_test = df_test[features].values
    Y_test = df_test[class_name].values
    X_test = mms.transform(X_test)

    # print(np.any(np.isinf(X_test)), np.any(np.isnan(X_test)), np.all(np.isfinite(X_test)))
    # print(np.any(np.isinf(Y_test)), np.any(np.isnan(Y_test)), np.all(np.isfinite(Y_test)))

    k_neighbors = min(5, np.unique(Y_test, return_counts=True)[1][1] - 1)
    if k_neighbors > 1:
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_test_bal, Y_test_bal = sm.fit_resample(X_test, Y_test)
    else:
        X_test_bal, Y_test_bal = X_test, Y_test

    if clf_type == 'DT':
        clf = DecisionTreeClassifier()
    elif clf_type == 'RF':
        clf = RandomForestClassifier()
    elif clf_type == 'AB':
        clf = AdaBoostClassifier()
    elif clf_type == 'ET':
        clf = ExtraTreesClassifier()
    else:
        raise Exception('Unknown classifier %s' % clf_type)

    rs = RandomizedSearchCV(clf, param_distributions=params[clf_type], n_iter=100, cv=5,
                            scoring=scoring, iid=False, n_jobs=5, verbose=1)

    if oversampling:
        rs.fit(X_train_bal, Y_train_bal)
    else:
        rs.fit(X_train, Y_train)

    clf = rs.best_estimator_
    pickle.dump(clf, open(path + 'crash_prediction_%s_%s_%s_%s_%s_%s.pickle' % (
        area, index, feature_type, clf_type, scoring, oversampling), 'wb'))

    Y_pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    # print('train')
    # print('accuracy', accuracy_train)
    # print(classification_report(Y_pred_train, Y_train))

    Y_pred_train_bal = clf.predict(X_train_bal)
    accuracy_train_bal = accuracy_score(Y_pred_train_bal, Y_train_bal)
    # print('train bal')
    # print('accuracy bal', accuracy_train_bal)
    # print(classification_report(Y_pred_train_bal, Y_train_bal))

    Y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(Y_pred_test, Y_test)
    # print('test')
    # print('accuracy', accuracy_test)
    # print(classification_report(Y_pred_test, Y_test))

    Y_pred_test_bal = clf.predict(X_test_bal)
    accuracy_test_bal = accuracy_score(Y_pred_test_bal, Y_test_bal)
    # print('test bal')
    # print('accuracy bal', accuracy_test)
    # print(classification_report(Y_pred_test_bal, Y_test_bal))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    features_ranking = {
        'indices': indices.tolist(),
        'features': features,
        'importances': importances.tolist(),
    }

    evaluation = {
        'area': area,
        'index': index,
        'train_crash': int(np.unique(Y_train, return_counts=True)[1][1]),
        'test_crash': int(np.unique(Y_test, return_counts=True)[1][1]),
        'clf': clf_type,
        'feature_type': feature_type,
        'scoring': scoring,
        'oversampling': oversampling,
        'train_accuracy': accuracy_train,
        'train_report': classification_report(Y_pred_train, Y_train, output_dict=True),
        'train_accuracy_bal': accuracy_train_bal,
        'train_report_bal': classification_report(Y_pred_train_bal, Y_train_bal, output_dict=True),
        'test_accuracy': accuracy_test,
        'test_report': classification_report(Y_pred_test, Y_test, output_dict=True),
        'test_accuracy_bal': accuracy_test_bal,
        'test_report_bal': classification_report(Y_pred_test_bal, Y_test_bal, output_dict=True),
        'features_ranking': features_ranking,
    }

    evaluation_file = open(path + 'crash_prediction_%s_%s_%s.json' % (area, index, feature_type), 'a')
    json_str = '%s\n' % json.dumps(evaluation)
    evaluation_file.write(json_str)
    evaluation_file.close()

    Y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
    Y_pred_proba_test = clf.predict_proba(X_test)[:, 1]

    df_train['crash_risk'] = Y_pred_proba_train
    df_test['crash_risk'] = Y_pred_proba_test

    crash_risk = {uid: v['crash_risk'] for uid, v in df_train[['crash_risk']].to_dict('index').items()}
    # crash_risk.update({'%s_test' % uid:
    # v['crash_risk'] for uid, v in df_test[['crash_risk']].to_dict('index').items()})
    cr_file = open(path + 'crash_risk_%s_%s_%s_%s_%s_%s.json' % (
        area, index, feature_type, clf_type, scoring, oversampling), 'w')
    json.dump(crash_risk, cr_file)
    cr_file.close()


def main():
    area = sys.argv[1]
    overwrite = str2bool(sys.argv[2])

    feature_type = 'teic'  #'t' 'e' 'i' 'c' 'teic' 'tei' 'tec' 'ic'
    class_name = 'crash'
    random_state = 0

    path_traintest = cfg.store_files["path_traintest"]
    path_dataset = cfg.store_files["path_dataset"]
    path_eval = cfg.store_files["path_eval"]

    for sel_index in ['0','1','2','3','4','5','6']:  #sel index of periods defined in crash_prediction_service.py
        print("sel_index ", sel_index)
        if overwrite:
            output_filename = path_eval + 'crash_prediction_%s_%s_%s.json' % (area, sel_index, feature_type)
            if os.path.exists(output_filename):
                os.remove(output_filename)

        filenames = dict()
        for filename in listdir(path_traintest):
            if area in filename:
                index = filename[filename.find('.csv.gz') - 1]
                if index not in filenames:
                    filenames[index] = dict()
                if 'train' in filename:
                    filenames[index]['train'] = filename
                else:
                    filenames[index]['test'] = filename

        features_names = json.load(open(path_dataset + 'features_names.json', 'r'))
        features_map = {'t': 'traj', 'e': 'evnt', 'i': 'imn', 'c': 'col'}

        features = list()
        for ft in feature_type:
            if ft in features_map:
                features.extend(features_names[features_map[ft]])

        print(filenames)
        # for index, fn in filenames.items():
        fn = filenames[sel_index]
        for clf_type in ['RF', 'DT', 'AB', 'ET']:
            for scoring in ['recall', 'f1']:
                for oversampling in [1, 0]:
                    print(datetime.datetime.now(), area, sel_index, feature_type, clf_type, scoring, oversampling)
                    print("reading file %s" % path_traintest + fn['train'])
                    df_train = pd.read_csv(path_traintest + fn['train'])
                    df_train.set_index('uid', inplace=True)
                    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_train.fillna(0, inplace=True)
                    df_train = df_train.reset_index().drop_duplicates(subset='uid', keep='first').set_index('uid')
                    # print(len(df_train.index.drop_duplicates(keep='first')))
                    # df_train = df_train.loc[df_train.index.drop_duplicates(keep='first')]
                    print("reading file %s" % path_traintest + fn['test'])
                    df_test = pd.read_csv(path_traintest + fn['test'])
                    df_test.set_index('uid', inplace=True)
                    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_test.fillna(0, inplace=True)
                    df_test = df_test.reset_index().drop_duplicates(subset='uid', keep='first').set_index('uid')
                    # print(df_train.isnull().sum()[df_train.isnull().sum() > 0])
                    # print(df_test.isnull().sum()[df_test.isnull().sum() > 0])
                    # print(len(df_train[df_train.isin([np.nan, np.inf, -np.inf]).any(1)]))
                    # print(len(df_test[df_test.isin([np.nan, np.inf, -np.inf]).any(1)]))

                    # features = [col for col in df_train.columns if col != class_name]
                    train_test_classifier(df_train, df_test, features, class_name, clf_type, scoring, oversampling,
                                          path_eval, area, sel_index, feature_type, random_state)
            #         break
            #     break
            # break


if __name__ == "__main__":
    main()