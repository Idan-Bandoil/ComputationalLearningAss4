import pandas as pd
from Algorithms.WeightedVotingClassifier import WeightedVotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def mfmw_new(X, y, k=100, m=10):
    '''
    mfmw algorithm
    
    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe with all samples
    y : pandas.Series
        Series with labels of all samples
    k : int
        Number of top_features to select
    m : int
        Number of features each filter outputs
    min_features : int
        Minimum number of top_features to select
    min_corr : float
        Minimum correlation between top_features
    wrong_thresh : float
        Threshold for wrong classifications
    indecisive_thresh : float
        Threshold for indecisive classifications

    Returns
    -------
    top_features : list
        List with scores of top_features
    '''

    df = pd.DataFrame(X.copy())
    features = df.columns
    # add y as a column to df
    df['class'] = y
    training_samples, test_samples = train_test_split(df, test_size=0.2, shuffle=True, random_state=10)

    # calculate corrcoef between each feature and the class
    corr = training_samples.corr()['class'].abs().sort_values(ascending=False)
    # calculate signal to noise ratio for each feature
    snr = ((training_samples.mean() / training_samples.std()) ** 2).sort_values(ascending=False)
    # remove class from corr and snr
    snr = snr.drop('class')
    corr = corr[corr.index != 'class']    

    def filter():
        # features sorted by corr, snr in descending order
        corr_features = list(corr.index)
        snr_features = list(snr.index)
        # m feature names with highest corrcoef, snr
        top_corr_features = corr[corr_features[:m]].index
        top_snr_features = snr[snr_features[:m]].index
        top_features = top_corr_features.append(top_snr_features).unique()
        # calculate mean between snr and corr for each feature in top_features
        mean_score = {feature: (snr[feature] + corr[feature]) / 2 for feature in features}
        # filter features with low correlation
        return top_features, mean_score

    def average_accuracy(X_train, y_train, X_test, y_test):
        knn, svm, wv = KNeighborsClassifier(n_neighbors=3, n_jobs=-1), SVC(class_weight='balanced'), WeightedVotingClassifier(snr)
        # fit KNN, SVM and DecisionTree with these features only
        knn.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        wv.fit(X_train, y_train)
        # return average accuracy of all classifiers
        return (knn.score(X_test, y_test) + svm.score(X_test, y_test) + wv.score(X_test, y_test)) / 3

    def wrapper(top_features, score):
        # divide training samples into X, y
        X_train, y_train = training_samples.iloc[:, :-1], training_samples.iloc[:, -1]
        X_test, y_test = test_samples.iloc[:, :-1] , test_samples.iloc[:, -1]
        best_subset, best_avg_accuracy = [], 0

        # for every feature of top_features, train classifiers with these features only and classify test samples
        for _ in range(len(top_features)):
            features_left = set(top_features) - set(best_subset)
            sorted_features = sorted(features_left, key=lambda feature: score[feature], reverse=True)
            chosen_feature = None
            iter_without_improvement = 0
            for feature in sorted_features:
                current_X_train, current_X_test = X_train[best_subset + [feature]], X_test[best_subset + [feature]]
                avg_accuracy = average_accuracy(current_X_train, y_train, current_X_test, y_test)
                # if current subset of features is better than previous best subset, update chosen_feature
                if avg_accuracy > best_avg_accuracy:
                    chosen_feature = feature
                    iter_without_improvement = 0
                    best_avg_accuracy = avg_accuracy
                if avg_accuracy == 1:
                    return [1 if feature in best_subset else score[feature] / sum(score) for feature in features]
                iter_without_improvement += 1
                if iter_without_improvement is not None and iter_without_improvement >= 10:
                    break
            if chosen_feature is not None:
                best_subset.append(chosen_feature)
            else:
                best_subset.append(sorted_features[0])
            if len(best_subset) >= k:
                return [1 if feature in best_subset else score[feature] / sum(score) for feature in features]
        return [1 if feature in best_subset else score[feature] / sum(score.values()) for feature in features]

    top_features, score = filter()
    result = wrapper(top_features, score)
    return result, None
