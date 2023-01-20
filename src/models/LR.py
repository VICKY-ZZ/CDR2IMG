from sklearn.linear_model import LogisticRegression
from src.models.deprecated.scoring import test_scoring,val_scoring
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.utils import evaluate_results,min_max_scale

import numpy as np
#
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
        print()
        print('Testing test:{} || val:{} || train:{}'.format(test_size[0], (1-test_size[0])*test_size[1], (1-test_size[0])*(1-test_size[1])))
        train_data, test_data = train_test_split(data, test_size=test_size[0], random_state=0, stratify=data['label'])

        # 对数据进行归一化处理
        new_train_data, new_test_data = min_max_scale(train_data, test_data)

        # 训练模型
        log_reg = LogisticRegression(penalty='l2', C=100,max_iter=2000)
        y_test_true = new_test_data['label']

        # 验证模型
        train_X, val_X, train_y, val_y = train_test_split(new_train_data.drop(columns=["phone_no_m", "label"]),
                                                          new_train_data["label"].values, test_size=test_size[1],random_state=0)
        # print('train_y:{}:{}:{}'.format(np.sum(train_y==1),np.sum(train_y==0), len(train_y)))
        # print('val_y:{}:{}:{}'.format(np.sum(val_y==1),np.sum(val_y==0),len(val_y)))
        # print('test_y:{}:{}:{}'.format(np.sum(y_test_true==1),np.sum(y_test_true==0),len(y_test_true)))
        #Train
        log_reg.fit(train_X, train_y)

        #Validation
        val_y_predict = log_reg.predict_proba(val_X)
        print('LR: Validation result:')
        evaluate_results(val_y, val_y_predict)    # 预测

        print('-----------------------------------------------------------------------------------')

        #Test
        new_test_data.drop(columns=["phone_no_m", "label"])
        y_test_predict = log_reg.predict_proba(new_test_data.drop(columns=["phone_no_m", "label"]))
        print('LR: Test result:')
        evaluate_results(y_test_true, y_test_predict)
