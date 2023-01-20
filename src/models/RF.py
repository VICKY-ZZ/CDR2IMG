from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.utils import evaluate_results,min_max_scale


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
        print('Testing test:{} || val:{} || train:{}'.format(test_size[0], (1 - test_size[0]) * test_size[1],
                                                             (1 - test_size[0]) * (1 - test_size[1])))
        train_data, test_data = train_test_split(data, test_size=test_size[0], random_state=0, stratify=data['label'])
        # 对数据进行归一化处理
        new_train_data, new_test_data = min_max_scale(train_data, test_data)

        # 保存test标签
        y_test_true = new_test_data['label']
        # 划分训练集、验证集
        train_X, val_X, train_y, val_y = train_test_split(new_train_data.drop(columns=["phone_no_m", "label"]),new_train_data["label"].values,test_size=test_size[1], random_state=0)


        # 1. 首先对n_estimators进行网格搜索：
        # param_test1 = {'n_estimators': [1, 10, 50, 120, 160, 200, 250]}
        # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
        #                                                          min_samples_leaf=20, max_depth=8, max_features='sqrt',
        #                                                          random_state=10),
        #                         param_grid=param_test1, scoring='roc_auc', cv=5)
        # gsearch1.fit(train_X, train_y)
        # print(gsearch1.best_params_)
        # best_n_estimators = 50
        #
        # # 2. 接着我们对决策树最大深度max_depth进行网格搜索。
        # param_test2 = {'max_depth': [1, 2, 3, 5, 7, 9, 10, 11, 12, 13]}
        # gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_n_estimators, min_samples_split=100,
        #                                                          min_samples_leaf=20, max_features='sqrt', oob_score=True,
        #                                                          random_state=10),
        #                         param_grid=param_test2, scoring='roc_auc', cv=5)
        # gsearch2.fit(train_X, train_y)
        # print(gsearch2.best_params_)
        # best_max_depth = 11
        #
        # # 3. 接着我们对决策树内部节点再划分所需最小样本数min_samples_split进行网格搜索。
        # param_test3 = {'min_samples_split': [45,46,47,48,49,50,51,52,53,54,55]}
        # gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth,
        #                                                          min_samples_leaf=20, max_features='sqrt', oob_score=True,
        #                                                          random_state=10),
        #                         param_grid=param_test3, scoring='roc_auc', cv=5)
        # gsearch3.fit(train_X, train_y)
        # print(gsearch3.best_params_)
        # best_min_samples_split = 51

        # rf1 = RandomForestClassifier(n_estimators=best_n_estimators,
        #                              max_depth=best_max_depth,
        #                              min_samples_split=best_min_samples_split, min_samples_leaf=20,
        #                              max_features='sqrt', oob_score=True, random_state=10)
        # rf1.fit(train_X, train_y)
        # val_y_pre = rf1.predict(val_X)
        # # val_y_prob_1 = rf1.predict_proba(val_X)[:, 1]
        # # 0.8882303132938189
        # print(accuracy_score(val_y, val_y_pre))
        # # 0.8216597821502454
        # print(roc_auc_score(val_y, val_y_pre))

        # # 4、这一次同时对所有的参数进行网格搜索
        # param_test_all = {
        #     'n_estimators': [50, 120, 160, 200, 250],
        #     'max_depth': [1, 2, 3, 5, 7, 9, 11, 13],
        #     'min_samples_split': [50, 80, 100, 120, 150, 180, 200, 300],
        #     'min_samples_leaf': [10, 20, 30, 40, 50, 100],
        #     'max_features': [3, 5, 7, 9, 11]
        # }
        # gsearch = GridSearchCV(estimator=RandomForestClassifier(oob_score=True, random_state=10),
        #                         param_grid=param_test_all, scoring='roc_auc', cv=5)
        # gsearch.fit(train_X, train_y)
        # print(gsearch.best_params_)



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
        evaluate_results(y_test_true,y_test_predict)

