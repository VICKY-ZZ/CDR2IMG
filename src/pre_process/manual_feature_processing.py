import numpy as np
import pandas as pd
from scipy.stats import stats
def get_voc_feat(df):
    print('开始处理数据')
    df["start_datetime"] = pd.to_datetime(df['start_datetime'])
    df["hour"] = df['start_datetime'].dt.hour
    df["day"] = df['start_datetime'].dt.day
    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')
    # 对话人数和对话次数
    print('     正在计算通话次数和通话人数')
    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(opposite_count="count", opposite_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    """主叫通话
    """
    print('正在处理通话类型为主叫的电话信息：')
    print('     正在计算imeis个数')
    df_call = df[df["calltype_id"] == 1].copy()
    tmp = df_call.groupby("phone_no_m")["imei_m"].agg(voccalltype1="count", imeis="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    print('     正在计算主叫占比')
    phone_no_m["voc_calltype1"] = phone_no_m["voccalltype1"] / phone_no_m["opposite_count"]
    # print('     正在计算所在城市和区县个数')
    # tmp = df_call.groupby("phone_no_m")["city_name"].agg(city_name_call="nunique")
    # phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    # tmp = df_call.groupby("phone_no_m")["county_name"].agg(county_name_call="nunique")
    # phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    print('     正在计算通话类型个数')
    tmp = df.groupby("phone_no_m")["calltype_id"].agg(calltype_id_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    """和固定通话者的对话统计
    """
    print('正在统计通话交互行为信息:')

    tmp = df.groupby(["phone_no_m", "opposite_no_m"])["call_dur"].agg(count="count", sum="sum")
    print('     正在统计和固定通话者的通话次数信息')
    phone2opposite = tmp.groupby("phone_no_m")["count"].agg(phone2opposite_mean="mean"
                                                            , phone2opposite_median="median"
                                                            , phone2opposite_max="max"
                                                            , phone2opposite_min="min"
                                                            , phone2opposite_var="var"
                                                            , phone2opposite_skew="skew"
                                                            , phone2opposite_sem="sem"
                                                            , phone2opposite_std="std"
                                                            , phone2opposite_quantile="quantile"
                                                            )

    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")
    print('     正在统计和固定通话者的通话总时长信息')
    phone2opposite = tmp.groupby("phone_no_m")["sum"].agg(phone2oppo_sum_mean="mean"
                                                          , phone2oppo_sum_median="median"
                                                          , phone2oppo_sum_max="max"
                                                          , phone2oppo_sum_min="min"
                                                          , phone2oppo_sum_var="var"
                                                          , phone2oppo_sum_skew="skew"
                                                          , phone2oppo_sum_sem="sem"
                                                          , phone2oppo_sum_std="std"
                                                          , phone2oppo_sum_quantile="quantile"
                                                          )

    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")

    """通话时间长短统计
    """
    print('     正在统计和固定通话者的每次通话时长信息')
    tmp = df.groupby("phone_no_m")["call_dur"].agg(call_dur_mean="mean"
                                                   , call_dur_median="median"
                                                   , call_dur_max="max"
                                                   , call_dur_min="min"
                                                   , call_dur_var="var"
                                                   , call_dur_skew="skew"
                                                   , call_dur_sem="sem"
                                                   , call_dur_std="std"
                                                   , call_dur_quantile="quantile"
                                                   )
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["city_name"].agg(city_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["county_name"].agg(county_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["calltype_id"].agg(calltype_id_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    """通话时间点偏好
    """
    print('正在处理通话时间偏好信息：')
    print('     正在计算每日最常通话时间点，及在该时间点通话次数，通话时间分布')
    tmp = df.groupby("phone_no_m")["hour"].agg(voc_hour_mode=lambda x: stats.mode(x)[0][0],
                                               voc_hour_mode_count=lambda x: stats.mode(x)[1][0],
                                               voc_hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    print('     正在计算每月最常通话日期，及在该日期通话次数，通话时间分布')
    tmp = df.groupby("phone_no_m")["day"].agg(voc_day_mode=lambda x: stats.mode(x)[0][0],
                                              voc_day_mode_count=lambda x: stats.mode(x)[1][0],
                                              voc_day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    phone_no_m.fillna(0, inplace=True)
    return phone_no_m

def get_sms_feats(df):

    print(df.head())
    df['request_datetime'] = pd.to_datetime(df['request_datetime'])
    df["hour"] = df['request_datetime'].dt.hour
    df["day"] = df['request_datetime'].dt.day


    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')

    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(sms_count="count", sms_nunique="nunique")
    tmp["sms_rate"] = tmp["sms_count"] / tmp["sms_nunique"]
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    calltype2 = df[df["calltype_id"] == 2].copy()
    calltype2 = calltype2.groupby("phone_no_m")["calltype_id"].agg(calltype_2="count")
    phone_no_m = phone_no_m.merge(calltype2, on="phone_no_m", how="left")
    phone_no_m["calltype_rate"] = phone_no_m["calltype_2"] / phone_no_m["sms_count"]

    tmp = df.groupby("phone_no_m")["hour"].agg(hour_mode=lambda x: stats.mode(x)[0][0],
                                               hour_mode_count=lambda x: stats.mode(x)[1][0],
                                               hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["day"].agg(day_mode=lambda x: stats.mode(x)[0][0],
                                              day_mode_count=lambda x: stats.mode(x)[1][0],
                                              day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    return phone_no_m
def get_app_feats(df):
    print(df.head())
    print(df["busi_name"].value_counts())
    phones_app = df[["phone_no_m"]].copy()
    phones_app = phones_app.drop_duplicates(subset=['phone_no_m'], keep='last')
    tmp = df.groupby("phone_no_m")["busi_name"].agg(busi_count="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["flow"].agg(flow_mean="mean",
                                               flow_median="median",
                                               flow_min="min",
                                               flow_max="max",
                                               flow_var="var",
                                               flow_sum="sum")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["month_id"].agg(month_ids="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    phones_app["flow_month"] = phones_app["flow_sum"] / phones_app["month_ids"]

    return phones_app

def get_user_feats(df):
    print(df.head())
    phones_app = df[["phone_no_m"]].copy()
    phones_app = phones_app.drop_duplicates(subset=['phone_no_m'], keep='last')

    phones_app['arpu_mean'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].mean(axis=1)

    phones_app['arpu_var'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].var(axis=1)
    phones_app['arpu_max'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].max(axis=1)
    phones_app['arpu_min'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].min(axis=1)
    phones_app['arpu_median'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].median(axis=1)
    phones_app['arpu_sum'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].sum(axis=1)
    phones_app['arpu_skew'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].skew(axis=1)

    phones_app['arpu_sem'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].sem(axis=1)
    phones_app['arpu_quantile'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].quantile(axis=1)


    return phones_app


def feats():
    train_voc = pd.read_csv(path + 'train_voc.csv', )
    train_voc_feat = get_voc_feat(train_voc)
    train_voc_feat.to_csv(path + "tmp/train_voc_feat.csv", index=False)

    train_app = pd.read_csv(path + 'train_app.csv', )
    train_app_feat = get_app_feats(train_app)
    train_app_feat.to_csv(path + "tmp/train_app_feat.csv", index=False)

    train_sms = pd.read_csv(path + 'train_sms.csv', )
    train_sms_feat = get_sms_feats(train_sms)
    train_sms_feat.to_csv(path + "tmp/train_sms_feat.csv", index=False)

    train_user = pd.read_csv(path + 'train_user.csv', )
    train_user_feat = get_user_feats(train_user)
    train_user_feat.to_csv(path + "tmp/train_user_feat.csv", index=False)
    print('feat extraction succeed!')

def merge_feat(path_feat,df):
    df_feat=pd.DataFrame(pd.read_csv(path_feat))
    return df.merge(df_feat,on='phone_no_m',how='left')

def feat_merge():
    df_user = pd.DataFrame(pd.read_csv(path + 'train_user.csv'))[["phone_no_m"]].copy()

    # new_user = merge_feat(path+'tmp/train_voc_feat.csv', df_user)

    # new_user = merge_feat(path+'tmp/train_sms_feat.csv', new_user)
    new_user = merge_feat(path+'tmp/train_sms_feat.csv', df_user)


    new_user = merge_feat(path+'tmp/train_app_feat.csv', new_user)

    new_user = merge_feat(path+'tmp/train_user_feat.csv', new_user)

    train_user = pd.DataFrame(pd.read_csv(path+'train_user.csv'))
    new_user = new_user.merge(train_user.loc[:,['phone_no_m','label']],on='phone_no_m',how='left')

    new_user.to_csv("../all_feat_without_voc_with_label.csv", index=False)
#只提取、生成voc特征
def voc_feats():
# 读取用户基本信息表
    train_user = pd.read_csv('../../data/train_user.csv', usecols=['phone_no_m', 'label'])
# 读取用户通话信息表
    train_voc = pd.read_csv('../../data/train_voc.csv')
# 获取用户通话特征
    train_voc_feat = get_voc_feat(train_voc)
# 与用户基本信息表合并
    train_voc_feat = train_voc_feat.merge(train_user, on="phone_no_m", how="left")
# 保存至voc_feat.csv
    train_voc_feat.to_csv("../voc_feat_with_county.csv", index=False)


path = '../../data/'
feat_merge()