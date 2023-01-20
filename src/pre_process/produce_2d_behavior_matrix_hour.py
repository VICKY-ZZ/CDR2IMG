# -*- coding: utf-8 -*-
# ！！！这个版本用二维矩阵——24*244,更正了填充方法。
import os
import gc
import time
import datetime
import numpy as np
import pandas as pd
import math
import random
from sklearn.utils import shuffle



# 获取通话时间特征
def get_voc_feat(df):
    print('正在处理时间列。。。')
    df["start_datetime"] = pd.to_datetime(df['start_datetime'])
    df["year"] = df['start_datetime'].dt.year
    df["month"] = df['start_datetime'].dt.month
    df["day"] = df['start_datetime'].dt.day
    df["hour"] = df['start_datetime'].dt.hour
    df['minute'] = df['start_datetime'].dt.minute
    print('--------------已添加时间列！-----------')
    return df

# 提取每个电话号码数据，分别生成文件//形成矩阵
def distract_phone_no_m(df):
    listType = df['phone_no_m'].unique()
    fraud_matrix =[]
    non_fraud_matrix=[]
    f=1
    n=1
 
    for j in range(len(listType)):
        print('--------------------------------------------------------------------------------')
        print('正在处理第'+str(j+1)+'个电话号码')
        # 查找【phone_no_m列含有第j个号码的行】，即处理第j个号码
        data_i = df[df['phone_no_m'].isin([listType[j]])]
        # 获取label
        if data_i.iloc[0]['label'] == 1:
            fraud = 1
        else:
            fraud = 0

        if fraud==1:
            print('正在处理第'+str(f)+'个诈骗电话')
            call_matrix = one_phone_number(data_i)
            f+=1
            fraud_matrix.append(call_matrix)
        elif fraud==0:
            print('正在处理第'+str(n)+'个非诈骗电话')
            call_matrix = one_phone_number(data_i)
            non_fraud_matrix.append(call_matrix)
            n+=1
    fraud_matrix =np.array(fraud_matrix)
    non_fraud_matrix = np.array(non_fraud_matrix)
    print('最终诈骗矩阵形状：')
    print(fraud_matrix.shape)
    print('最终非诈骗矩阵形状:')
    print(non_fraud_matrix.shape)
    print('正在粘贴诈骗矩阵与非诈骗矩阵')
    X = np.concatenate((fraud_matrix,non_fraud_matrix),axis=0)
    print('正在保存X矩阵')
    np.save(file='../../data/processed_data/hour/X_new_coarse_2.npy', arr=X)
    print('保存X矩阵完毕！')
# 先尝试形成一个number的矩阵
def one_phone_number(df):
    if df.iloc[0]['label'] == 1:
        fraud = 1
    else:
        fraud = 0
    # nf为填数矩阵，根据nf中opposite_count字段的值对矩阵进行填数
    # 求opposite_count这列的数量
    nf =df.groupby(["phone_no_m","opposite_no_m",'calltype_id'])['calltype_id'].agg(opposite_count="count")
    df = df.merge(nf,on=["phone_no_m","opposite_no_m",'calltype_id'])


    d1 = datetime.datetime(2019,8,1)   # 第一个日期
    d2 = datetime.datetime(2020,3,31)   # 第二个日期
    interval = d2 - d1                   # 两日期差距
    days = interval.days+1  #矩阵的列
    hours = 24
    #     初始化矩阵
    call_matrix = np.zeros((hours, days))
    for m in range(df.shape[0]):
        # 将主叫置为正值，被叫置为负值
        if df.iloc[m]['calltype_id'] == 1:
            pass
        elif df.iloc[m]['calltype_id'] == 2:
            df.loc[m,'call_dur'] = -df.iloc[m]['call_dur']
    # 计算每小时通话时间
    tmp = df.groupby(["year", 'day','month','hour'])["call_dur"].agg(call_dur_sum='sum')
    # print(tmp)
    df = df.merge(tmp,how='left',on=["year", 'day','month','hour'])

    for m in range(df.shape[0]):
        year = df.iloc[m]['year']
        day = df.iloc[m]['day']
        month = df.iloc[m]['month']
        hour = df.iloc[m]['hour']
        call_dur_sum = df.iloc[m]['call_dur_sum']
        d3 = datetime.datetime(year, month, day)
        # 计算当前计算的日期距离初始日期的位置
        interval = d3 - d1
        column_index = interval.days
        # 没必要做除法计算
        # if (call_dur_sum / 60) / 60 > 0:
        #     call_matrix[hour,column_index] = 1
        # if (call_dur_sum / 60) / 60 < 0:
        #     call_matrix[hour,column_index] = -1
        if call_dur_sum > 0:
            call_matrix[hour,column_index] = 1
        elif call_dur_sum < 0:
            call_matrix[hour,column_index] = -1
    call_matrix=np.array(call_matrix)

    return call_matrix

#

# 读取原始数据，添加时间列
def process_data():
    # 获取原始通话信息
    voc_ori = pd.read_csv('../../data/train_voc.csv', low_memory=False)
    # 获取通话时间信息
    voc = get_voc_feat(voc_ori)
    # 删除不需要的列,添加label
    print('正在删除不需要的列,添加label')
    # 获取用户信息
    user = pd.read_csv('../../data/train_user.csv', usecols=[0, 12])
    drop_r = ['city_name','county_name']
    voc.drop(drop_r, axis=1, inplace=True)
    voc = voc.merge(user, on="phone_no_m", how="left")
    print('已合并voc和user，目前voc带有label')

    distract_phone_no_m(voc)
def generate_y(y_path):
    y_fraud=np.ones((1892),dtype='int')
    y_non_fraud = np.zeros((4133),dtype='int')
    y = np.concatenate((y_fraud,y_non_fraud),axis=0)
    # # np.save(file='../data/processed/final_matrix/X.npy', arr=X)
    np.save(file=y_path, arr=y)

# 提取特征，生成x
# process_data()
# 生成y
# y_path = './processed_data/y_new.npy'
# generate_y(y_path)



# # # ---------------------把X.npy和y.npy粘上，并打乱顺序，以便使用迭代器---------------------------------
X= np.load('../../data/processed_data/hour/X_new_coarse_2.npy')
y =np.load('../../data/processed_data/hour/y_new.npy')
X, y = shuffle(X, y, random_state=0)
np.save('../../data/processed_data/hour/X_shuffled.npy', X)
np.save('../../data/processed_data/hour/y_shuffled.npy', y)
# for i in range(len(X)):
#     np.save('./coarse_2/X_'+str(i)+'.npy',arr=X[i,:,:])
#     np.save('./coarse_2/y_'+str(i)+'.npy',arr=y[i])

