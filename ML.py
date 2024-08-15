import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from imblearn.metrics import specificity_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict
import os

from FeatureExtraction import feature_headers


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
    testing_indicies = X_test.index
    scalar = StandardScaler().fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

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


def plot_predict(X_train, X_test, y_train, y_test, data, MSI_validation_folder='', MSS_validation_folder='', title='', validation=False):
    # performing gridsearch and prediction
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=22)
    param_grid = {
    'n_estimators' : [int(n) for n in np.linspace(start=1000, stop=5000, num=1000)],
    'max_features' :  ['auto', 'sqrt'],
    'max_depth' : [int(n) for n in np.linspace(start=10, stop=110, num=11)],
    'min_samples_split' : [2, 5, 7],
    'min_samples_leaf': [1, 2, 4],
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
        auroc, ci_l, ci_u, ax = validation_sets(MSI_validation_folder, MSS_validation_folder, clf, ax)
    else:
        auroc, ci_l, ci_u = [], [], []

    # confidence intervals and AUROC from initial prediction
    ci_l.append(confidence_lower)
    ci_u.append(confidence_upper)
    auroc.append(og_auroc)

    ax.legend(labels=['AUROC:' + str('%.3f' % np.mean(auroc)), 'Confidence Interval: [{:0.3f} - {:0.3f}]'.format(np.min(ci_l), np.max(ci_u)),
                      'Balanced Accuracy:' + str('%.3f' % balanced_accuracy_score(y_test, y_pred)),
                      'Specificty: ' + str('%.3f' % specificity_score(y_test, y_pred))], handlelength=0)
    plt.show()

    feature_importance(clf, X_train, X_test, y_train, y_test, data, title=title)

    return y_pred


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
        
    return auroc, ci_lower, ci_upper, ax


def feature_importance(model, X_train, X_test, y_train, y_test, data, title=''):
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Measure correlation amongst features
    corr = spearmanr(data).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # Convert the correlation matrix to a distance matrix before performing hierarchical clustering
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    X_train_sel = (X_train.T[selected_features]).T
    X_test_sel = (X_test.T[selected_features]).T

    clf_sel = RandomForestClassifier(n_estimators=1000, random_state=22)
    clf_sel.fit(X_train_sel, y_train)
    print(
        "Baseline accuracy on test data with features removed:"
        f" {clf_sel.score(X_test_sel, y_test):.2}"
    )

    _, ax = plt.subplots(figsize=(10, 8))
    plot_permutation_importance(clf_sel, X_test_sel, y_test, ax, data)
    ax.set_title("Permutation Importances on subset of features\n" + (title))
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
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


def per_patient(X_test, y_pred, y_test, test_ind, title='', validation=False):
    # Uses Kather's labels, which don't differentiate between MSI-H and MSI-L, unlike the GDC labels
    substring = 'TCGA'
    patients = {}
    for i in range(len(y_pred)):
        sample_name_indices = test_ind[i].find(substring)
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
    ax.set(title=title, xlabel='FPR', ylabel='TPR')
    ax.legend(labels=['AUROC: ' + str('%.3f' % roc_auc_score(y_test_label, y_pred_ratio)), 'Balanced Accuracy: ' + str('%.3f' % balanced_accuracy_score(y_test_label, y_pred_label))],
              handlelength=0)


