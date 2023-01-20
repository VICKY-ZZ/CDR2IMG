from sklearn.linear_model import LogisticRegression
from src.models.deprecated.scoring import test_scoring,val_scoring
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.utils import evaluate_results,min_max_scale,min_max_scale_plus_val,load_data_with_diffent_P_N_Ratio_for_ML


#
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
        # print('Testing test:{} || val:{} || train:{}'.format(test_size[0], (1-test_size[0])*test_size[1], (1-test_size[0])*(1-test_size[1])))
        # train_data, test_data = train_test_split(data, test_size=test_size[0], random_state=0, stratify=data['label'])
        train, val, test = load_data_with_diffent_P_N_Ratio_for_ML('../voc_feat_sorted.csv', dataset)
        # 对数据进行归一化处理
        new_train_data, new_val_data, new_test_data = min_max_scale_plus_val(train, val, test)

        # 训练模型
        log_reg = LogisticRegression(penalty='l2', C=100,max_iter=2000)

        train_y = new_train_data['label']
        train_X = new_train_data.drop(columns=["phone_no_m", "label"])
        val_y = new_val_data['label']
        val_X = new_val_data.drop(columns=["phone_no_m", "label"])
        test_y = new_test_data['label']
        test_X = new_test_data.drop(columns=["phone_no_m", "label"])
        # 验证模型
        # train_X, val_X, train_y, val_y = train_test_split(new_train_data.drop(columns=["phone_no_m", "label"]),
        #                                                   new_train_data["label"].values, test_size=test_size[1],random_state=0)
        #Train
        log_reg.fit(train_X, train_y)

        #Validation
        val_y_predict = log_reg.predict_proba(val_X)
        print('LR: Validation result:')
        evaluate_results(val_y, val_y_predict)    # 预测

        print('-----------------------------------------------------------------------------------')

        #Test
        # new_test_data.drop(columns=["phone_no_m", "label"])
        y_test_predict = log_reg.predict_proba(test_X)
        print('LR: Test result:')
        evaluate_results(test_y, y_test_predict)
