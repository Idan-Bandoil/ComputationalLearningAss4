import numpy as np
from Algorithms.WOA import WOA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from skfeature.function.similarity_based.reliefF import reliefF
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import pearsonr

def tri_stage(X, y, k=100, m=100, search_agents=50):
    '''
    Tri-Stage Feature Selection Algorithm

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.
    y : numpy array of shape (n_samples,)
        The class labels.
    k : int, optional
        Number of top_features to select. The default is 100.
    m : int, optional
        Number of features each filter outputs. The default is 100.
    search_agents : int, optional
        Number of search agents. The default is 50.

    Returns 
    -------
    top_features : list
        List with scores of top_features
    '''

    unique, counts = np.unique(y, return_counts=True)
    for i in range(len(unique)):
        if counts[i] == 1:
            idx = np.where(y == unique[i])
            X = np.delete(X, idx, axis=0)
            y = np.delete(y, idx, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=10, stratify=y)
    label_encoder = LabelEncoder().fit(y)
    y_train_encoded, y_test_encoded = label_encoder.transform(y_train), label_encoder.transform(y_test)

    def XV(feature_col, y):
        # get subset of feature_col grouped by class
        class_subsets = [feature_col[y == c] for c in np.unique(y)]
        # calculate variance for each class
        variances = [np.var(s) for s in class_subsets]
        return sum(variances)

    
    def filter():
        # select m sorted_features with the highest mutual information with the class
        mi_scores = mutual_info_classif(X, y)
        mi_scores = mi_scores.argsort()[-m:]
        # select m sorted_features with the highest relief score
        relief_scores = reliefF(X, y)
        relief_scores = relief_scores.argsort()[-m:]
        # select m sorted_features with the best chi2 score
        pipe = Pipeline([('scaler', MinMaxScaler(feature_range=(0, 1))), ('chi2', SelectKBest(chi2, k=min(m, X.shape[1])))]).fit(X, y)
        chi2_scores = pipe['chi2'].scores_
        chi2_scores = chi2_scores.argsort()[-m:]
        # select m sorted_features with the highest XV score
        XV_scores = np.array([XV(X[:, i], y) for i in range(X.shape[1])])
        XV_scores = XV_scores.argsort()[-m:]

        return np.unique(np.concatenate((mi_scores, relief_scores, chi2_scores, XV_scores)))

    
    def naive_classifiers(filtered_features):
        # Run KNN, NB, SVM classifiers on each feature and sort them in descending order by average accuracy
        knn, nb, svm = KNeighborsClassifier(n_neighbors=3, n_jobs=-1), GaussianNB(), SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced')
        classifiers = [knn, nb, svm]
        # for each feature, run the classifiers on the samples restricted to that feature
        feature_accuracies = []
        for i in filtered_features:
            averaged_accuracy = 0
            X_train_i = X_train[:, i].reshape(-1, 1)
            X_test_i = X_test[:, i].reshape(-1, 1)
            for clf in classifiers:
                clf.fit(X_train_i, y_train)
                averaged_accuracy += clf.score(X_test_i, y_test)
            averaged_accuracy /= len(classifiers)
            feature_accuracies.append((i, averaged_accuracy))
        sorted_feature_accuracies = sorted(feature_accuracies, key=lambda pair: pair[1], reverse=True)
        # return the features sorted by accuracy
        return [pair[0] for pair in sorted_feature_accuracies]


    def get_top_by_XGB(features):
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8,
            colsample_bytree=0.8, objective='multi:softmax', num_class=len(np.unique(y)), nthread=-1, seed=10)
        # get X_train reduced to features
        X_train_reduced = X_train[:, features]
        xgb.fit(X_train_reduced, y_train_encoded)
        # get feature importances
        feature_importances = xgb.feature_importances_
        # get features sorted by importance in descending order
        sorted_features_indexes = np.argsort(feature_importances)[::-1]
        sorted_features = np.array(features)[sorted_features_indexes.astype(int)]
        accuracy, prev_accuracy = 0, None
        low, high = 0, len(sorted_features) - 1
        while low < high:
            k = (low + high) // 2
            curr_features = sorted_features[:k]
            X_train_reduced, X_test_reduced = X_train[:, curr_features], X_test[:, curr_features]
            xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8, 
            colsample_bytree=0.8, objective='binary:logistic', nthread=-1, seed=10)
            xgb.fit(X_train_reduced, y_train_encoded)
            prev_accuracy = accuracy
            accuracy = xgb.score(X_test_reduced, y_test_encoded)
            if accuracy > prev_accuracy:
                low = k + 1
            else:
                high = k - 1
        return sorted_features[:k]

    # PHASE 1:
    filtered_features = filter()
    if len(filtered_features) >= k:
        return [1 if i in filtered_features else 0 for i in range(X.shape[1])], None
    sorted_features = naive_classifiers(filtered_features)
    top_k_features = get_top_by_XGB(sorted_features)
    if len(top_k_features) >= k:
        return [1 if i in top_k_features else 0 for i in range(X.shape[1])], None
    
    # PHASE 2:
    # create a pearson correlation matrix for the top k features and the class
    pearson_corr = np.zeros((len(top_k_features) + 1, len(top_k_features) + 1))
    for i in range(len(top_k_features)):
        # add the correlation between the feature and y to the pearson correlation matrix
        pearson_corr[i, len(top_k_features)] = pearsonr(X[:, top_k_features[i]], y)[0]
        pearson_corr[len(top_k_features), i] = pearson_corr[i, len(top_k_features)]
        # add the correlation between the feature and the other features to the pearson correlation matrix
        for j in range(i, len(top_k_features)):
            pearson_corr[i, j] = pearsonr(X[:, top_k_features[i]], X[:, top_k_features[j]])[0]
            pearson_corr[j, i] = pearson_corr[i, j]
        
    # get rid of the highly correlated features
    i, n = 0, len(top_k_features)
    while i < n:
        j = i + 1
        while j < n:
            if pearson_corr[i, j] > 0.9:
                # remove the feature with the lower correlation to the class
                if pearson_corr[i, len(top_k_features)] < pearson_corr[j, len(top_k_features)]:
                    # remove i from the numpy.array of top k features and update the loop
                    top_k_features = np.delete(top_k_features, i)
                else:
                    top_k_features = np.delete(top_k_features, j)
                n -= 1
            j += 1
        i += 1

    if len(top_k_features) >= k:
        return [1 if i in top_k_features else 0 for i in range(X.shape[1])], None

    top_j_features = get_top_by_XGB(top_k_features)

    if len(top_j_features) >= k:
        return [1 if i in top_j_features else 0 for i in range(X.shape[1])], None

    # get X_train restricted to the top j features
    X_train_top_j = X_train[:, top_j_features]
    # PHASE 3:
    features = WOA(X_train_top_j, y_train, n=search_agents)
    # return list containing 1 for each features in features and 0 for each feature not in features
    return [1 if i in features else 0 for i in range(X.shape[1])], None

