from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
# from scoring import test_scoring,val_scoring
from src.models.utils import evaluate_results,min_max_scale


def SVM_model(kernel='linear',C=100.0, gamma='auto', decision_function_shape='ovr', cache_size=500,class_weight='balanced'):
    print('--------------------------------KERNERL={}--------------------------------'.format(kernel))
    # 训练
    # 1/model = svm.SVC(C=100.0, kernel='linear', gamma='auto', decision_function_shape='ovr', cache_size=500)
    model = svm.SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape=decision_function_shape, cache_size = cache_size, class_weight = class_weight)
    model.fit(train_X, train_y)
    val_y_predict = model.predict(val_X)
    print('SVM-{}: Validation result:'.format(kernel))
    evaluate_results(val_y, val_y_predict)
#     预测
    new_test_data.drop(columns=["phone_no_m", "label"])
    y_test_predict = model.predict(new_test_data.drop(columns=["phone_no_m", "label"]))
    print('SVM-{}: Test result:'.format(kernel))
    evaluate_results(y_test_true, y_test_predict)
    print()

if __name__ == '__main__':
    # 获取训练集、测试集数据
    data = pd.read_csv('../all_feat_with_label.csv')
    data.dropna(axis=0, how='any', inplace=True)
    for test_size in [(0.2, 0.25), (0.2, 0.5), (0.6, 0.5), (0.8, 0.5)]:
        # (A,B),A为测试集比例，B为验证集和训练集划分比例
        # 测试集0.2剩80%，0.2验证，0.6训练
        # 测试集0.2剩60%，0.4验证，0.4训练
        # 测试集0.6剩40%，0.2验证，0.2训练
        # 测试集0.8剩20%，0.1验证，0.1训练
        print('Testing test:{} || val:{} || train:{}'.format(test_size[0], (1-test_size[0])*test_size[1], (1-test_size[0])*(1-test_size[1])))
        train_data, test_data = train_test_split(data, test_size=test_size[0],random_state=0,stratify=data['label'])
        # 对数据进行归一化处理
        new_train_data, new_test_data = min_max_scale(train_data, test_data)
        # # 保存test标签
        y_test_true = new_test_data['label']
        # 划分训练集、验证集
        train_X, val_X, train_y, val_y = train_test_split(new_train_data.drop(columns=["phone_no_m", "label"]),
                                                          new_train_data["label"].values, test_size=test_size[1],random_state=0)
        SVM_model(kernel='linear')
        SVM_model(kernel='poly')
        SVM_model(kernel='rbf')
        SVM_model(kernel='sigmoid')






