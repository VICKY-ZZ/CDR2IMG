import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import  matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

def data_split(X,Y,path):
    test_split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8,random_state=0)
    # val_split = StratifiedShuffleSplit(n_splits=1,test_size=0.25, train_size=0.75, random_state=0)
    for train_index, test_index in test_split.split(X,Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    np.savez_compressed(path+'/train',X_train=X_train, Y_train=Y_train)
    np.savez_compressed(path+'/test', X_test=X_test, Y_test=Y_test)

def ori_vs_fft(ori_matrix,i ,fraud =1):
    #0的转变
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(ori_matrix, cmap=plt.cm.Blues)
    plt.title("{}--x_{}".format(fraud,i))
    # plt.show()
    ori_matrix_fft = np.fft.fft(ori_matrix, axis=0)
    ori_matrix_fft_real = np.real(ori_matrix_fft)
    plt.subplot(2, 1, 2)
    plt.imshow(ori_matrix_fft_real, cmap=plt.cm.Blues)
    plt.title("{}--x_{}_fft_real".format(fraud,i))
    # plt.show()
    plt.savefig('../../fft_pics/min/{}/{}--x_{}'.format(fraud,fraud,i))