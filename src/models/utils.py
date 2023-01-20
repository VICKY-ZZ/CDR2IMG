import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,roc_auc_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data(path,fft=False):
    train = np.load(path + 'train.npz')
    test = np.load(path + 'test.npz')
    X_train, Y_train = train['X_train'], train['Y_train']
    X_test, Y_test = test['X_test'], test['Y_test']
    del train, test
    if fft:
        for i in range(len(X_train)):
            ori_matrix_fft = np.fft.fft(X_train[i, :, :], axis=0)
            X_train[i, :, :] = np.real(ori_matrix_fft)
        for i in range(len(X_test)):
            ori_matrix_fft = np.fft.fft(X_test[i, :, :], axis=0)
            X_test[i, :, :] = np.real(ori_matrix_fft)
    y_train = pd.DataFrame(Y_train, columns=['Y_train'])
    print(y_train.value_counts())
    print('-----------------------------------------')
    print('X_train.shape:{}'.format(X_train.shape))
    print('X_test.shape:{}'.format(X_test.shape))
    X_train = data_reshape(X_train)
    X_test = data_reshape(X_test)
    return X_train, X_test, Y_train, Y_test

def load_data_with_diffent_P_N_Ratio(path,dataset,fft=False):
    all = np.load(path)
    X_all, Y_all = all['X'], all['Y']
    del all
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size[0], random_state=0, stratify=Y)
    # y_fraud = np.ones((1892), dtype='int')
    # y_non_fraud = np.zeros((4133), dtype='int')
    if dataset=='SC1':
        X_train = np.concatenate((X_all[:300], X_all[1892:(1892+300)]), axis=0)
        Y_train = np.concatenate((Y_all[:300], Y_all[1892:(1892+300)]), axis=0)
        X_val = np.concatenate((X_all[300:600], X_all[(1892+300):(1892+600)]), axis=0)
        Y_val = np.concatenate((Y_all[300:600], Y_all[(1892 + 300):(1892 + 600)]), axis=0)
        X_test = np.concatenate((X_all[600:900], X_all[(1892+3300):(1892+3600)]), axis=0)
        Y_test = np.concatenate((Y_all[600:900], Y_all[(1892+3300):(1892+3600)]), axis=0)
    elif dataset=='SC5':
        X_train = np.concatenate((X_all[:300], X_all[1892:(1892 + 1500)]), axis=0)
        Y_train = np.concatenate((Y_all[:300], Y_all[1892:(1892 + 1500)]), axis=0)
        X_val = np.concatenate((X_all[300:600], X_all[(1892 + 1500):(1892 + 1800)]), axis=0)
        Y_val = np.concatenate((Y_all[300:600], Y_all[(1892 + 1500):(1892 + 1800)]), axis=0)
        X_test = np.concatenate((X_all[600:900], X_all[(1892+3300):(1892+3600)]), axis=0)
        Y_test = np.concatenate((Y_all[600:900], Y_all[(1892+3300):(1892+3600)]), axis=0)
    elif dataset=='SC10':
        X_train = np.concatenate((X_all[:300], X_all[1892:(1892 + 3000)]), axis=0)
        Y_train = np.concatenate((Y_all[:300], Y_all[1892:(1892 + 3000)]), axis=0)
        X_val = np.concatenate((X_all[300:600], X_all[(1892 + 3000):(1892 + 3300)]), axis=0)
        Y_val = np.concatenate((Y_all[300:600], Y_all[(1892 + 3000):(1892 + 3300)]), axis=0)
        X_test = np.concatenate((X_all[600:900], X_all[(1892 + 3300):(1892 + 3600)]), axis=0)
        Y_test = np.concatenate((Y_all[600:900], Y_all[(1892 + 3300):(1892 + 3600)]), axis=0)
    elif dataset=='TC1':
        X_train = np.concatenate((X_all[:300], X_all[1892:(1892 + 300)]), axis=0)
        Y_train = np.concatenate((Y_all[:300], Y_all[1892:(1892 + 300)]), axis=0)
        X_val = np.concatenate((X_all[300:600], X_all[(1892 + 300):(1892 + 600)]), axis=0)
        Y_val = np.concatenate((Y_all[300:600], Y_all[(1892 + 300):(1892 + 600)]), axis=0)
        X_test = np.concatenate((X_all[600:900], X_all[(1892 + 600):(1892 + 900)]), axis=0)
        Y_test = np.concatenate((Y_all[600:900], Y_all[(1892 + 600):(1892 + 900)]), axis=0)
    elif dataset=='TC5':
        X_train = np.concatenate((X_all[:300], X_all[1892:(1892 + 300)]), axis=0)
        Y_train = np.concatenate((Y_all[:300], Y_all[1892:(1892 + 300)]), axis=0)
        X_val = np.concatenate((X_all[300:600], X_all[(1892 + 300):(1892 + 600)]), axis=0)
        Y_val = np.concatenate((Y_all[300:600], Y_all[(1892 + 300):(1892 + 600)]), axis=0)
        X_test = np.concatenate((X_all[600:900], X_all[(1892 + 600):(1892 + 2100)]), axis=0)
        Y_test = np.concatenate((Y_all[600:900], Y_all[(1892 + 600):(1892 + 2100)]), axis=0)
    elif dataset=='TC10':
        X_train = np.concatenate((X_all[:300], X_all[1892:(1892 + 300)]), axis=0)
        Y_train = np.concatenate((Y_all[:300], Y_all[1892:(1892 + 300)]), axis=0)
        X_val = np.concatenate((X_all[300:600], X_all[(1892 + 300):(1892 + 600)]), axis=0)
        Y_val = np.concatenate((Y_all[300:600], Y_all[(1892 + 300):(1892 + 600)]), axis=0)
        X_test = np.concatenate((X_all[600:900], X_all[(1892 + 600):(1892 + 3600)]), axis=0)
        Y_test = np.concatenate((Y_all[600:900], Y_all[(1892 + 600):(1892 + 3600)]), axis=0)
    if fft:
        for i in range(len(X_train)):
            ori_matrix_fft = np.fft.fft(X_train[i, :, :], axis=0)
            X_train[i, :, :] = np.real(ori_matrix_fft)
        for i in range(len(X_val)):
            ori_matrix_fft = np.fft.fft(X_val[i, :, :], axis=0)
            X_val[i, :, :] = np.real(ori_matrix_fft)
        for i in range(len(X_test)):
            ori_matrix_fft = np.fft.fft(X_test[i, :, :], axis=0)
            X_test[i, :, :] = np.real(ori_matrix_fft)
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_val, Y_val = shuffle(X_val, Y_val, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    print('-----------------------------------------')
    print('X_train.shape:{}'.format(X_train.shape))
    print('X_val.shape:{}'.format(X_val.shape))
    print('X_test.shape:{}'.format(X_test.shape))
    X_train = data_reshape(X_train)
    X_val = data_reshape(X_val)
    X_test = data_reshape(X_test)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def load_data_with_diffent_P_N_Ratio_for_ML(path, dataset):
    all = pd.read_csv(path)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size[0], random_state=0, stratify=Y)
    # 1892:4133
    if dataset=='SC1':
        train = pd.concat((all[:300], all[1892:(1892+300)]), axis=0)
        val = pd.concat((all[300:600], all[(1892+300):(1892+600)]), axis=0)
        # test = pd.concat((all[600:900], all[(1892+600):(1892+900)]), axis=0)
        test = pd.concat((all[600:900], all[(1892 + 3300):(1892 + 3600)]), axis=0)

    elif dataset=='SC5':
        train = pd.concat((all[:300], all[1892:(1892 + 1500)]), axis=0)
        val = pd.concat((all[300:600], all[(1892 + 1500):(1892 + 1800)]), axis=0)
        # test = pd.concat((all[600:900], all[(1892 + 1800):(1892 + 2100)]), axis=0)

        test = pd.concat((all[600:900], all[(1892 + 3300):(1892 + 3600)]), axis=0)

        # train = pd.concat((all[:300], all[3000:(3000 + 1500)]), axis=0)
        # val = pd.concat((all[300:600], all[(3000 + 1500):(3000 + 1800)]), axis=0)
        # test = pd.concat((all[600:900], all[(3000 + 1800):(3000 + 2100)]), axis=0)
    elif dataset=='SC10':
        train = pd.concat((all[:300], all[1892:(1892 + 3000)]), axis=0)
        val = pd.concat((all[300:600], all[(1892 + 3000):(1892 + 3300)]), axis=0)
        test = pd.concat((all[600:900], all[(1892 + 3300):(1892 + 3600)]), axis=0)
    elif dataset=='TC1':
        train = pd.concat((all[:300], all[1892:(1892+300)]), axis=0)
        val = pd.concat((all[300:600], all[(1892+300):(1892+600)]), axis=0)
        test = pd.concat((all[600:900], all[(1892+600):(1892+900)]), axis=0)
    elif dataset=='TC5':
        train = pd.concat((all[:300], all[1892:(1892+300)]), axis=0)
        val = pd.concat((all[300:600], all[(1892+300):(1892+600)]), axis=0)
        test = pd.concat((all[600:900], all[(1892+600):(1892+2100)]), axis=0)
    elif dataset=='TC10':
        train = pd.concat((all[:300], all[1892:(1892+300)]), axis=0)
        val = pd.concat((all[300:600], all[(1892+300):(1892+600)]), axis=0)
        test = pd.concat((all[600:900], all[(1892+600):(1892+3600)]), axis=0)
    train = shuffle(train, random_state=0)
    val = shuffle(val, random_state=0)
    test = shuffle(test, random_state=0)

    print('-----------------------------------------')
    print('train.shape:{}'.format(train.shape))
    print('val.shape:{}'.format(val.shape))
    print('test.shape:{}'.format(test.shape))
    return train, val, test
def load_data_with_diffent_ratio(path,test_size,fft=False):
    all = np.load(path)
    X, Y = all['X'], all['Y']
    del all
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size[0], random_state=0, stratify=Y)
    if fft:
        for i in range(len(X_train)):
            ori_matrix_fft = np.fft.fft(X_train[i, :, :], axis=0)
            X_train[i, :, :] = np.real(ori_matrix_fft)
        for i in range(len(X_test)):
            ori_matrix_fft = np.fft.fft(X_test[i, :, :], axis=0)
            X_test[i, :, :] = np.real(ori_matrix_fft)
    y_train = pd.DataFrame(Y_train, columns=['Y_train'])
    print(y_train.value_counts())
    print('-----------------------------------------')
    print('X_train.shape:{}'.format(X_train.shape))
    print('X_test.shape:{}'.format(X_test.shape))
    X_train = data_reshape(X_train)
    X_test = data_reshape(X_test)
    return X_train, X_test, Y_train, Y_test
def data_reshape(X_train):
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
    return X_train

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_results(Y_val_true, Y_val_pred):
    if Y_val_pred.ndim==1:
        Y_val_pred_classes = Y_val_pred
    else:
        Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
    confusion_test_mtx = confusion_matrix(Y_val_true, Y_val_pred_classes)
    # plot the confusion matrix

    # plot_confusion_matrix(confusion_test_mtx, classes=range(2))

    precision, recall, f1, _ = precision_recall_fscore_support(Y_val_true, Y_val_pred_classes, average="macro")
    accuracy = accuracy_score(Y_val_true, Y_val_pred_classes)
    if Y_val_pred.ndim==1:
        auc = roc_auc_score(Y_val_true, Y_val_pred, average='macro')
    else:
        auc = roc_auc_score(Y_val_true, Y_val_pred[:, 1], average='macro')
    print("Accuracy:%.4f || AUC: %.4f || Precision: %.4f || Recall: %.4f || F1: %.4f" % (accuracy, auc, precision, recall, f1))

def display_errors(Y_val_pred_classes, Y_val_true,Y_val_pred,X_val):
    errors = (Y_val_pred_classes - Y_val_true != 0)
    Y_pred_classes_errors = Y_val_pred_classes[errors]
    Y_pred_errors = Y_val_pred[errors]
    Y_true_errors = Y_val_true[errors]
    X_val_errors = X_val[errors]
    # Probabilities of the wrong predicted numbers
    Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)
    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    # Top 6 errors
    most_important_errors = sorted_dela_errors[-6:]
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = most_important_errors[n]
            ax[row,col].imshow(X_val_errors[error])
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(Y_pred_classes_errors[error],Y_true_errors[error]))
            n += 1
    plt.show()

def min_max_scale_plus_val(train_data, val_data, test_data, exclude=None):
    # 默认不归一化手机号码和标签
    if exclude is None:
        exclude = ["phone_no_m", "label", 'is_train']

    # 合并训练集、测试集数据。用于后面的特征归一化处理
    train_data["is_train"], test_data["is_train"], val_data['is_train'] = [1] * len(train_data), [0] * len(test_data), [2] * len(val_data)
    data = train_data.append(test_data).reset_index(drop=True)
    data = data.append(val_data).reset_index(drop=True)
    # 筛选要归一化的特征列表
    columns = list(data.columns)
    for column_to_drop in exclude:
        if column_to_drop in columns:
            columns.pop(columns.index(column_to_drop))

    # 对数据进行归一化处理
    mm = MinMaxScaler()
    data.loc[:, columns] = mm.fit_transform(data.loc[:, columns])

    return (data[data["is_train"] == 1].drop(columns=["is_train"]).copy(),
            data[data["is_train"] == 2].drop(columns=["is_train"]).copy(),
            data[data["is_train"] == 0].drop(columns=["is_train"]).copy())
# 将训练集数据、测试集数据进行归一化处理
def min_max_scale(train_data, test_data, exclude=None):
    # 默认不归一化手机号码和标签
    if exclude is None:
        exclude = ["phone_no_m", "label"]

    # 合并训练集、测试集数据。用于后面的特征归一化处理
    train_data["is_train"], test_data["is_train"] = [1] * len(train_data), [0] * len(test_data)
    data = train_data.append(test_data).reset_index(drop=True)
    # 筛选要归一化的特征列表
    columns = list(data.columns)
    for column_to_drop in exclude:
        if column_to_drop in columns:
            columns.pop(columns.index(column_to_drop))

    # 对数据进行归一化处理
    # 训练集和测试集一起归一化???
    mm = MinMaxScaler()
    data.loc[:, columns] = mm.fit_transform(data.loc[:, columns])

    return (data[data["is_train"] == 1].drop(columns=["is_train"]).copy(),
            data[data["is_train"] == 0].drop(columns=["is_train"]).copy())

def data_split_and_save_for_2d_matrix(path,X,Y):
    test_split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8,random_state=0)
    # val_split = StratifiedShuffleSplit(n_splits=1,test_size=0.25, train_size=0.75, random_state=0)
    for train_index, test_index in test_split.split(X,Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    np.savez_compressed(path,X_train=X_train, Y_train=Y_train)
    np.savez_compressed(path, X_test=X_test, Y_test=Y_test)



from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    # "font.size": 15,
    # "mathtext.fontset":'stix',
}
rcParams.update(config)
# 设置图片大小
fig = plt.figure(figsize=(14,8))

# # 解决中文显示问题
# font = {
#     'family':'SimHei',
#     'weight':'bold',
#     'size':12
# }
# matplotlib.rc('font',**font)

def line_chart(a,b,c,d):
    plt.figure(figsize=(14, 8))
    length = len(a)
    x = [0.05*i for i in range(length)]
    x_l = ["{:.2f}".format(i) for i in x]

    # if ylabel=='Recall' or ylabel=='F1':
    #     y = [0.70, 0.80, 0.90]
    #     y_l = [str(i) for i in y]
    # else:
    #     y = [0.85, 0.90, 0.95]
    #     y_l = [str(i) for i in y]

    # 画图：设置折线属性
    plt.plot(x, a, 'o-', color='#F6BE61', label='AUC')
    plt.plot(x, b, 's-', color='#439DC0', label='Precision')
    plt.plot(x, c, '*-', color='#F65675', label='F1')
    plt.plot(x, d, 'x-', color='#84A59E', label='Recall')


    # 设置横纵坐标的刻度值
    plt.xticks(ticks=x,labels=x_l)
    # plt.yticks(ticks=y,labels=y_l)

    # 给图片加网格
    # plt.grid()
    # 设置横纵坐标的标签
    plt.ylabel('Score')
    plt.xlabel('Width Shift Range')
    plt.legend(loc="lower left")

    # # 保存图片
    plt.savefig('./AUG/Augmentation{}.png'.format(length))
    plt.savefig('./AUG/Augmentation{}.eps'.format(length), dpi=600, format='eps')
    plt.savefig('./AUG/Augmentation{}.tif'.format(length), dpi=600, format='tif')
    plt.savefig('./AUG/Augmentation{}.svg'.format(length), dpi=600, format='svg')
    plt.show()





