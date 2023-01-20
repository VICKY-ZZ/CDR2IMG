import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import os
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from src.models.utils import evaluate_results,min_max_scale

if __name__ == '__main__':
    batch_size = 8
    num_classes = 2
    epochs = 40
    conv_kernel_size = 6
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
        optimizer = Adam(lr=0.0001, decay=1e-6)
        test_X = new_test_data.drop(columns=["phone_no_m", "label"])
        y_test_true = new_test_data['label']
        # 验证模型
        train_X, val_X, train_y, val_y = train_test_split(new_train_data.drop(columns=["phone_no_m", "label"]),
                                                          new_train_data["label"].values, test_size=test_size[1],
                                                          random_state=0)

        x_train = (train_X.values.reshape(train_X.shape[0], train_X.shape[1],1)).astype('float32')
        x_val = (val_X.values.reshape(val_X.shape[0], val_X.shape[1],1)).astype('float32')
        x_test = (test_X.values.reshape(test_X.shape[0], test_X.shape[1],1)).astype('float32')

        # print('x_train shape:', x_train.shape)
        # print(x_train.shape[0], 'train samples')
        # print(x_val.shape[0], 'validation samples')
        # print(x_test.shape[0], 'test samples')
        # print(x_train.shape[1])
        model = Sequential()

        # ---改版
        model.add(Conv1D(64, conv_kernel_size, padding='same',input_shape=(x_train.shape[1], 1), activation='relu'))
        model.add(Conv1D(128, conv_kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))


        model.add(Conv1D(64, conv_kernel_size, padding='same',input_shape=(x_train.shape[1], 1), activation='relu'))
        model.add(Conv1D(128, conv_kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
# ----原版
# model.add(Conv1D(64, conv_kernel_size, padding='same',input_shape=(x_train.shape[1], 1), activation='relu'))
# model.add(Conv1D(128, conv_kernel_size, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
#
# model.add(Conv1D(64, conv_kernel_size, padding='same',input_shape=(x_train.shape[1], 1), activation='relu'))
# model.add(Conv1D(128, conv_kernel_size, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

        # model.summary()


        model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        model.fit(x_train, train_y, batch_size=batch_size, epochs=epochs, validation_data=(x_val, val_y), shuffle=True)
                  #callbacks = [rocauc])


        save_dir = os.path.join(os.getcwd(), 'saved_cnn_models')
        model_name = 'keras_cnn_trained_model_epochs'+str(epochs)+'.h5'


        val_y_predict = model.predict(val_X)
        # val_y_predict = [0 if i[0]>0.5 else 1 for i in val_y_predict]
        print('CNN_manual_feature: Validation result:')
        evaluate_results(val_y, val_y_predict)    # 预测

        # print(tmp_predict)
        test_y_predict = model.predict(x_test)
        # test_y_predict = [0 if i[0]>0.5 else 1 for i in test_y_predict]
        # print(predict)
        print('CNN_manual_feature: Test result:')
        evaluate_results(y_test_true, test_y_predict)


