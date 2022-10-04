import numpy as np
from itertools import product

class WeightedVotingClassifier:
    def __init__(self, snr, thresh=0.3):
        self.snr = snr
        self.thresh = thresh
        self.X = None
        self.y = None
        self.features = None
        self.classes = None
        self.binary = False
        self.means = {}

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.features = X.columns
        self.classes = np.unique(y)
        self.means = {g: {class_: 0 for class_ in self.classes} for g in self.features}
        if len(self.classes) == 2:
            self.binary = True
        # calculate mean of each feature for each class
        for feature, class_ in product(X, self.classes):
            self.means[feature][class_] = np.mean(X[feature][y == class_])

    def predict(self, X, epsilon=0.00000001):
        n = X.shape[0]
        if self.binary:
            V = np.array([[0, 0]] * n)
            for i, g in product(range(n), self.features):
                sample = X.iloc[i]
                x_g = sample[g]
                # get mean of feature for each class
                miu_1 = self.means[g][self.classes[0]]
                miu_2 = self.means[g][self.classes[1]]
                v_g = self.snr[g] * (x_g - (miu_1 + miu_2) / 2)
                if (miu_1 > miu_2 and v_g > 0) or (miu_1 < miu_2 and v_g < 0):
                    V[i][0] += v_g
                elif (miu_1 > miu_2 and v_g < 0) or (miu_1 < miu_2 and v_g > 0):
                    V[i][1] += v_g
            return np.array([self.classes[0] if v[0] > v[1] else self.classes[1] for v in V])
        else:
            class_votes = np.array([{class_: 0 for class_ in self.classes}] * n)
            for i, g in product(range(n), self.features):
                sample = X.iloc[i]
                x_g = sample[g]
                # get mean of feature for each class
                mius = [self.means[g][class_] for class_ in self.classes]
                # add to class votes the class which has the closest miu to x_g
                chosen_class_idx = np.argmin(np.abs(np.array(mius) - x_g))
                chosen_class = self.classes[chosen_class_idx]
                class_votes[i][chosen_class] += self.snr[g] / (np.abs(x_g - mius[chosen_class_idx]) + epsilon)
            return np.array([max(class_votes[i], key=class_votes[i].get) for i in range(n)])

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)
