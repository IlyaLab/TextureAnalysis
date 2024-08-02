import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import os

plt.rcParams.update({'font.size': 22})

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
    MSI_df = combine_subsets(MSI_folder)
    MSS_df = combine_subsets(MSS_folder)
    y_MSI = np.ones(len(MSI_df))
    y_MSS = np.zeros(len(MSS_df))
    X = pd.concat( (MSI_df, MSS_df), axis=0)
    y = np.concatenate((y_MSI, y_MSS))

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=22, shuffle=True)
    scalar = StandardScaler().fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

    return X_train, X_test, y_train, y_test


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


def plot_predict(X_train, X_test, y_train, y_test, MSI_validation_folder='', MSS_validation_folder='', title='', validation=False):
    # performing gridsearch and prediction
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=22)
    param_grid = {
    'n_estimators' : [1000],
    'bootstrap' : [True, False],
    }
    clf = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=kf).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # bootstrapping to get AUROC confidence internval
    confidence_upper, confidence_lower = bootstrapping(y_test, y_pred_proba, n_bootstraps=10000, rng_seed=22)

    # generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
    og_auroc = roc_auc_score(y_test, y_pred_proba[:,1])
    _, ax = plt.subplots(figsize=(10,8))
    ax.set(title=title, xlabel='FPR', ylabel='TPR',)
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], 'k--')

    if validation == True:
        auroc, ci_l, ci_u = validation_sets(MSI_validation_folder, MSS_validation_folder, clf, ax)
    else:
        auroc, ci_l, ci_u = [], [], []

    # confidence intervals and AUROC from initial prediction
    ci_l.append(confidence_lower)
    ci_u.append(confidence_upper)
    auroc.append(og_auroc)

    ax.legend(labels=['AUROC:' + str('%.2f' % np.mean(auroc)), 'Confidence Interval: [{:0.3f} - {:0.3}]'.format(np.min(ci_l), np.max(ci_u)) ], handlelength=0)
    plt.show()


def validation_sets(MSI_folder, MSS_folder, model, ax):
    # predicting on validation sets using trained model and plotting to same plot
    auroc = []
    ci_lower = []
    ci_upper = []
    for MSI_set, MSS_set in zip(os.listdir(MSI_folder), os.listdir(MSS_folder)):
        MSI_df = pd.read_csv(os.path.join(MSI_folder, MSI_set))
        MSS_df = pd.read_csv(os.path.join(MSS_folder, MSS_set))

        MSI_df.set_index('Tile', inplace=True)
        MSS_df.set_index('Tile', inplace=True)

        y_MSI = np.ones(len(MSI_df))
        y_MSS = np.zeros(len(MSS_df))
        X = pd.concat( (MSI_df, MSS_df), axis=0)
        y = np.concatenate((y_MSI, y_MSS))

        scalar = StandardScaler().fit(X)
        X = scalar.transform(X)

        y_pred_proba = model.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_pred_proba[:,1])
        score = roc_auc_score(y, y_pred_proba[:,1])
        auroc.append(score)

        confidence_upper, confidence_lower = bootstrapping(y, y_pred_proba, n_bootstraps=10000, rng_seed=22)
        ci_lower.append(confidence_lower)
        ci_upper.append(confidence_upper)

        ax.plot(fpr, tpr)
        
    return auroc, ci_lower, ci_upper