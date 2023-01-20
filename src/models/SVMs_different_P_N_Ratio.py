from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
# from scoring import test_scoring,val_scoring
from src.models.utils import evaluate_results,min_max_scale_plus_val,load_data_with_diffent_P_N_Ratio_for_ML


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
    evaluate_results(test_y, y_test_predict)
    print()

if __name__ == '__main__':
    # 获取训练集、测试集数据
    data = pd.read_csv('../all_feat_without_voc_with_label_sorted.csv')
    # data.dropna(axis=0, how='any', inplace=True)
    different_sets = ['SC1', 'SC5', 'SC10']
    # different_sets = ['TC1', 'TC5', 'TC10']
    for dataset in different_sets:
        print('Testing:{}'.format(dataset))
        # (A,B),A为测试集比例，B为验证集和训练集划分比例
        # 测试集0.2剩80%，0.2验证，0.6训练
        # 测试集0.2剩60%，0.4验证，0.4训练
        # 测试集0.6剩40%，0.2验证，0.2训练
        # 测试集0.8剩20%，0.1验证，0.1训练
        train, val, test = load_data_with_diffent_P_N_Ratio_for_ML('../voc_feat_sorted.csv', dataset)
        # 对数据进行归一化处理
        new_train_data, new_val_data, new_test_data = min_max_scale_plus_val(train, val, test)
        train_y = new_train_data['label']
        train_X = new_train_data.drop(columns=["phone_no_m", "label"])
        val_y = new_val_data['label']
        val_X = new_val_data.drop(columns=["phone_no_m", "label"])
        test_y = new_test_data['label']
        test_X = new_test_data.drop(columns=["phone_no_m", "label"])
        SVM_model(kernel='linear')
        SVM_model(kernel='poly')
        SVM_model(kernel='rbf')
        SVM_model(kernel='sigmoid')






