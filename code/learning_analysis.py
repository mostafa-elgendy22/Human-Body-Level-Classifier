import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy.signal import savgol_filter

def learning_analysis(model_architecture, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model_architecture, X=X, y=y,cv=10, 
                                                            train_sizes=np.linspace(0.01, 1.0, 100), n_jobs=2)

    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_mean = savgol_filter(train_mean, window_length=11, polyorder=3, mode="nearest")
    test_mean = savgol_filter(test_mean, window_length=11, polyorder=3, mode="nearest")

    plt.style.use('default')
    plt.plot(train_sizes, train_mean, label = 'train_acc', color = 'b')
    plt.plot(train_sizes, test_mean, label = 'test_acc', color = 'r')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title(f'Learning Curve of {type(model_architecture).__name__}', fontsize = 18, y = 1.03)
    plt.legend()
