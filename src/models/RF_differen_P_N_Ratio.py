from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.utils import evaluate_results, min_max_scale, load_data_with_diffent_P_N_Ratio_for_ML, \
    min_max_scale_plus_val

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
        print()
        train, val, test = load_data_with_diffent_P_N_Ratio_for_ML('../voc_feat_sorted.csv', dataset)
        # 对数据进行归一化处理
        new_train_data, new_val_data, new_test_data = min_max_scale_plus_val(train, val, test)

        train_y = new_train_data['label']
        train_X = new_train_data.drop(columns=["phone_no_m", "label"])
        val_y = new_val_data['label']
        val_X = new_val_data.drop(columns=["phone_no_m", "label"])
        test_y = new_test_data['label']
        test_X = new_test_data.drop(columns=["phone_no_m", "label"])
        # {'max_depth': 13, 'max_features': 9, 'min_samples_leaf': 10, 'min_samples_split': 50, 'n_estimators': 200}
        rf = RandomForestClassifier(oob_score=True, random_state=12,max_depth=13,max_features=9,min_samples_leaf=10,min_samples_split=50,n_estimators=200)
        #Training
        rf.fit(train_X, train_y)

        #Validation
        print('RF: Validation result:')
        val_y_pre = rf.predict_proba(val_X)
        evaluate_results(val_y, val_y_pre)

        print('-----------------------------------------------------------------------------------')

        #Test
        print('RF: Test result:')
        y_test_predict= rf.predict_proba(new_test_data.drop(columns=["phone_no_m", "label"]))
        evaluate_results(test_y,y_test_predict)

