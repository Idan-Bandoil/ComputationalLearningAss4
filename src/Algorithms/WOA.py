import numpy as np
from numpy.random import rand
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def calculate_error(x, k, X_train, y_train, X_test, y_test):    
    X_train_reduced = X_train[:, x == 1]
    X_test_Reduced = X_test[:, x == 1]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_reduced, y_train)
    error = 1 - knn.score(X_test_Reduced, y_test)

    return error


def fitness(x, k, X_train, y_train, X_test, y_test, alpha=0.99):
    # Number of selected features
    num_feat = np.sum(x == 1)
    # No feature selected
    if num_feat == 0:
        return 1
    else:
        error = calculate_error(x, k ,X_train, y_train, X_test, y_test)
        # Objective function
        return alpha * error + (1 - alpha) * (num_feat / len(x))


def to_bin(X, thresh, n, dimension):
    return np.array([[1 if X[i, d] > thresh else 0 for d in range(dimension)] for i in range(n)])


def enforce_boundary(x, lower_bound, upper_bound):
    return lower_bound if x < lower_bound else upper_bound if x > upper_bound else x


def WOA(X, y, k=5, n=50, max_iter=50, thresh=0.5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=10)
    
    # Dimension
    dimension = X.shape[1]
    upper_bound *= np.ones([1, dimension], dtype='float')
    lower_bound *= np.ones([1, dimension], dtype='float')
    
    # Initialize position 
    positions = np.zeros([n, dimension], dtype='float')
    for i, d in product(range(n), range(dimension)):
        positions[i, d] = lower_bound[0, d] + (upper_bound[0, d] - lower_bound[0, d]) * rand() 
    
    X_binary = to_bin(positions, thresh, n, dimension)
    
    # fitness at first iteration
    fitness = np.zeros([n, 1], dtype='float')
    best_position = np.zeros([1, dimension], dtype='float')
    best_fitness = float('inf')
    
    for i in range(n):
        fitness[i, 0] = fitness(X_binary[i, :], k, X_train, y_train, X_test, y_test)
        if fitness[i, 0] < best_fitness:
            best_position[0, :] = positions[i, :]
            best_fitness = fitness[i, 0]
        
    p = 1
    while p < max_iter:
        # Eq (8)
        s = 2 - p * (2 / max_iter)
        
        for i in range(n):
            # Eq (6), (7)
            K, J = 2 * s * rand() - s, 2 * rand()
            t = rand()
            if t < 0.5:
                if abs(K) < 1:
                    for d in range(dimension):
                        # Eq (5)
                        B = abs(J * best_position[0, d] - positions[i, d])
                        # Position update
                        positions[i, d] = best_position[0, d] - K * B
                        positions[i, d] = enforce_boundary(positions[i, d], lower_bound[0, d], upper_bound[0, d])
                
                else:
                    for d in range(dimension):
                        # Select a random whale
                        q = np.random.randint(low=0, high=n)
                        # Eq (12)
                        B = abs(J * positions[q, d] - positions[i, d])
                        # Position update
                        positions[i, d] = positions[q, d] - K * B
                        positions[i, d] = enforce_boundary(positions[i, d], lower_bound[0, d], upper_bound[0, d])
            
            else:
                l = 2 * rand() - 1
                for d in range(dimension):
                    dist = abs(best_position[0, d] - positions[i, d])
                    # Position update Eq (9)
                    positions[i, d] = dist * np.exp(l) * np.cos(2 * np.pi * l) + best_position[0, d] 
                    positions[i, d] = enforce_boundary(positions[i, d], lower_bound[0, d], upper_bound[0, d])
        
        # Binary conversion
        X_binary = to_bin(positions, thresh, n, dimension)
        
        # fitness
        for i in range(n):
            fitness[i, 0] = fitness(X_binary[i, :], k, X_train, y_train, X_test, y_test)
            if fitness[i, 0] < best_fitness:
                best_position[0, :] = positions[i, :]
                best_fitness = fitness[i, 0]
        
        p += 1            

    best_position_bin = to_bin(best_position, thresh, 1, dimension).reshape(dimension)
    features_num = np.asarray(range(0, dimension))    
    return features_num[best_position_bin == 1] # The selected features
    