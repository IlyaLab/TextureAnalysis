import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, precision_score, confusion_matrix
from imblearn.metrics import specificity_score
import xgboost as xgb # version 2.1.4
import torch
import os
import joblib

from FeatureExtraction import feature_headers
from sklearn.inspection import permutation_importance

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict


plt.rcParams.update({'font.size': 18})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combine_subsets(input_dir):
    # multiprocessing splits up the texture analysis results into separate csv files, so this combines them into a single one
    df = 0
    for file in os.listdir(input_dir):
        csv = pd.read_csv(os.path.join(input_dir, file))
        csv.set_index('Tile', inplace=True)
        try:
            df = pd.concat((df, csv), axis=0)
        except:
            df = csv
    return df


def preprocess_data(MSI_folder, MSS_folder):
    # assigns MSI and MSS labels based off of which folder they came from (ala Kather dataset)
    MSI_df, MSS_df = combine_subsets(MSI_folder), combine_subsets(MSS_folder)
    y_MSI, y_MSS = np.ones(len(MSI_df)), np.zeros(len(MSS_df))
    X = pd.concat( (MSI_df, MSS_df), axis=0)
    y = np.concatenate((y_MSI, y_MSS))

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=22, shuffle=True)
    testing_indicies = X_test.index
    scalar = StandardScaler().fit(X_train)
    X_train, X_test = scalar.transform(X_train), scalar.transform(X_test)

    if device.type == 'cuda':
        X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
        X = torch.tensor(X.values)

    return X_train, X_test, y_train, y_test, X, list(testing_indicies)


def bootstrapping(y_test, y_pred_proba, n_bootstraps = 10000, rng_seed=22):
    # boostrapping performed in order to find confidence interval of model
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_proba[:,1]), len(y_pred_proba[:,1]))
        if len(np.unique(y_test[indices])) < 2:
            continue

        score = roc_auc_score(y_test[indices], y_pred_proba[:,1][indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

    return confidence_upper, confidence_lower


def choose_ML(model, ML):
    if ML == 'XGBoost_models':
        if model == 'CRC':
            model = 'TCGA-CRC_model.ubj'
        elif model == 'STAD':
            model = 'TCGA-STAD_model.ubj'
        elif model == 'UCEC':
            model = 'TCGA-UCEC_model.ubj'
        else:
            model = 'model.ubj'
    else:
        if model == 'CRC':
            model = 'TCGA-CRC_model.pkl'
        elif model == 'STAD':
            model = 'TCGA-STAD_model.pkl'
        elif model == 'UCEC':
            model = 'TCGA-UCEC_model.pkl'
        else:
            model = 'model.pkl'

    return model


def plot_predict(X_train, X_test, y_train, y_test, data, MSI_validation_folder='', MSS_validation_folder='', title='', validation=False, model='', ML='XGBoost'):
    # performing gridsearch and prediction

    ML_folder = ML + '_models'

    model = choose_ML(model, ML_folder)

    if model not in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder)): 
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        # here we are going to be testing prediction with a few different models:
        if ML == 'XGBoost':

            '''XGBoost'''
            base_clf = xgb.XGBClassifier(device='cuda', booster='dart', n_jobs=-1, random_state=22)
            param_grid ={
                'n_estimators' : [100],
                'max_depth' : [3,5],
                'eta' : [0.1, 0.3],
                'sampling_method' : ['uniform', 'gradient_based'],
                'objective' : ['binary:logistic']
            }
        elif ML == 'RandomForest':

            '''RandomForest'''
            base_clf = RandomForestClassifier(n_jobs=-1, random_state=22)
            param_grid = {
                'n_estimators' : [1000],#[int(n) for n in np.linspace(start=1000, stop=5000, num=10)],
                'max_depth' : [None],
                'max_samples' : [0.5, 0.7, 1.0]
            }
        elif ML == 'SVM':

            '''SVM (SVC)'''
            base_clf = SVC(random_state=22)
            param_grid = {
                'C' : [0.5, 1.0],
                'kernel' : ['rbf', 'poly'],
                'degree' : [3, 4, 5],
                'class_weight' : ['balanced'],
                'shrinking' : [True, False]
            }
        elif ML == 'Regression':

            '''Logistic Regression'''
            base_clf = LogisticRegression(random_state=22)
            param_grid = {
                'penalty' : ['l1', 'l2'],
                'C' : [0.3, 0.5, 1.0],
                'solver' : ['liblinear']
            }

        clf = GridSearchCV(estimator=base_clf, param_grid=param_grid, cv=kf).fit(X_train, y_train)
        pretrained = False
        print('Training complete.')
    elif model in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder)):
        print(f'Loading trained model {model}.')
        if ML == 'XGBoost':
            clf = xgb.XGBClassifier(device=device)
            clf.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder, model))
        else:
            clf = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder, model))

        pretrained = True

    # save best training parameters on a hold out set
    best_params_df = pd.DataFrame([clf.best_params_])
    best_params_df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder, (title + '_best_params.csv')))

    results_df = pd.DataFrame(clf.cv_results_)
    results_df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder, (title + '_cv_results.csv')))

    y_pred = clf.predict(X_test)
    try:
        y_pred_proba = clf.predict_proba(X_test)
    except:
        # allows for models such as SVM without any issue
        y_pred_proba = y_pred
        temp = np.zeros_like(y_pred_proba)
        y_pred_proba = np.column_stack((temp, y_pred_proba))

    # bootstrapping to get AUROC confidence internval
    confidence_upper, confidence_lower = bootstrapping(y_test, y_pred_proba, n_bootstraps=10000, rng_seed=22)

    # generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
    og_auroc = roc_auc_score(y_test, y_pred_proba[:,1])
    _, ax = plt.subplots(figsize=(10,8))
    # ax.set(title=title, xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], 'k--')

    if validation == True:
        auroc, ci_l, ci_u, ax = validation_sets(MSI_validation_folder, MSS_validation_folder, clf, ax)
    else:
        auroc, ci_l, ci_u = [], [], []

    # locally save trained model
    if pretrained == False:
        if ML == 'XGBoost':
            clf.best_estimator_.save_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder, (title + '_model.ubj')))
        else:
            _ = joblib.dump(clf.best_estimator_, os.path.join(os.path.dirname(os.path.realpath(__file__)), ML_folder, (title + '_model.pkl')))


    # confidence intervals and AUROC from initial prediction
    ci_l.append(confidence_lower)
    ci_u.append(confidence_upper)
    auroc.append(og_auroc)

    # calc confusion matrix metrics
    cm = confusion_matrix(y_test, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    neg_pred_val = TN / (TN + FN)


    # ax.legend(labels=['AUC:' + str('%.3f' % np.mean(auroc)) + ' [{:0.3f} - {:0.3f}]'.format(np.min(ci_l), np.max(ci_u)),
    #                   'Balanced Accuracy:' + str('%.3f' % balanced_accuracy_score(y_test, y_pred)),
    #                   'Specificty: ' + str('%.3f' % specificity_score(y_test, y_pred)),
    #                   'Precision (PPV): ' + str('%.3f' % precision_score(y_test, y_pred)),
    #                   'NPV: '+ '{:0.3f}'.format(neg_pred_val[1])], handlelength=0)
    plt.show()
    # try:
    #     clf.best_estimator_
    #     feature_importance(clf.best_estimator_, data)
    # except:
    #     feature_importance(clf,data)
    

    return y_pred


def validation_sets(MSI_folder, MSS_folder, model, ax):
    # predicting on validation sets using trained model and plotting to same plot
    auroc = []
    ci_lower = []
    ci_upper = []
    for MSI_set, MSS_set in zip(os.listdir(MSI_folder), os.listdir(MSS_folder)):
        MSI_df, MSS_df = pd.read_csv(os.path.join(MSI_folder, MSI_set)), pd.read_csv(os.path.join(MSS_folder, MSS_set))

        MSI_df.set_index('Tile', inplace=True)
        MSS_df.set_index('Tile', inplace=True)

        y_MSI, y_MSS = np.ones(len(MSI_df)), np.zeros(len(MSS_df))
        X = pd.concat( (MSI_df, MSS_df), axis=0)
        y = np.concatenate((y_MSI, y_MSS))

        scalar = StandardScaler().fit(X)
        X = scalar.transform(X)

        if device.type == 'cuda':
            X = torch.from_numpy(X)

        try:
            y_pred_proba = model.predict_proba(X)
        except:
            # allows for usage of models like SVM
            y_pred_proba = model.predict(X)
            temp = np.zeros_like(y_pred_proba)
            y_pred_proba = np.column_stack((temp, y_pred_proba))
        
        fpr, tpr, _ = roc_curve(y, y_pred_proba[:,1])
        score = roc_auc_score(y, y_pred_proba[:,1])
        auroc.append(score)

        confidence_upper, confidence_lower = bootstrapping(y, y_pred_proba, n_bootstraps=10000, rng_seed=22)
        ci_lower.append(confidence_lower)
        ci_upper.append(confidence_upper)

        ax.plot(fpr, tpr)
        
    return auroc, ci_lower, ci_upper, ax


 #The feature_importance() functions are depreciated since switching to XGBoost as the ML model, instead of RandomForest.

def feature_importance(model, data):
    # plt.rcParams.update(plt.rcParamsDefault)

    try:
        model.get_booster().feature_names = feature_headers 
        feature_important = model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
        ax = data.loc[:,"score"].tail(10).plot(kind='barh', figsize = (20,10)) # plot top features

        ax.set_title("Feature Importance\n")
        ax.set_xlabel("Model Weight")
        ax.set_ylabel("Features")
    except:
        mdi_importances = pd.Series(
            model.feature_importances_, index=feature_headers
            ).sort_values(ascending=True)
        ax = mdi_importances.tail(5).plot(kind='barh', figsize = (20,10))
        ax.set_title("Random Forest Feature Importances (MDI)")
        ax.set_ylabel('Features')
    
    plt.show()


def plot_permutation_importance(clf, X_test, y_test, ax, data):
    result = permutation_importance(clf, X_test, y_test, n_repeats=100, random_state=22, n_jobs=-1)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=data.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


# NOTE: per_patient() currently only supports TCGA data, due to sample & tile labelling
def per_patient(y_pred, y_test, test_ind, title='', study='TCGA'):
    # Uses Kather's labels, which don't differentiate between MSI-H and MSI-L, unlike the GDC labels
    substring = study
    patients = {}
    for i in range(len(y_pred)):
        sample_name_indices = test_ind[i].find(substring)
        if study == 'TCGA':
            sample_name =  test_ind[i][sample_name_indices:sample_name_indices+12]

        df = pd.DataFrame(np.column_stack((test_ind[i], y_pred[i], y_test[i])), columns=['ID', 'pred', 'test'])

        if sample_name not in patients.keys():
            patients[sample_name] = df
        else:
            patients[sample_name] = pd.concat((patients[sample_name], df), axis=0)
    
    # Ratio of y_pred vs y_test to determine if patient was correctly predicted
    y_test_label, y_pred_ratio, y_pred_label = [], [], []
    for value in patients.values():
        y_test_label.append(np.mean(np.array(value['test']), dtype=float))
        y_pred_ratio.append(np.mean(np.array(value['pred']), dtype=float))
        if np.mean(np.array(value['pred']), dtype=float) <= 0.4: # the value given by MSIMantis to determine MSS vs MSI
            y_pred_label.append(0)
        elif np.mean(np.array(value['pred']), dtype=float) > 0.4:
            y_pred_label.append(1)
        
    _, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test_label, y_pred_ratio)
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], 'k--')
    # ax.set(title=title, xlabel='False Positive Rate', ylabel='True Positive Rate')
    # ax.legend(labels=['AUC: ' + str('%.3f' % roc_auc_score(y_test_label, y_pred_ratio)), 'Balanced Accuracy: ' + str('%.3f' % balanced_accuracy_score(y_test_label, y_pred_label))],
    #           handlelength=0)


