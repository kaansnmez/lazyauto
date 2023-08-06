import pandas as pd
import numpy as np
def datetime_simple(df_date):
    """
    Datetime column transform and seperate to day,month,year,week,dayofweek ... etc.

    Parameters
    ----------
    df_date : DataFrame / Series TYPE
        Datetime DataFrame/Series.

    Returns
    -------
    DataFrame TYPE
        
    """
    def is_spring(df):
        date = pd.to_datetime(df)
        return (date.month >= 2) & (date.month <= 4)
    def is_summer(df):
        date = pd.to_datetime(df)
        return (date.month >= 5) & (date.month <= 7)
    def is_autumn(df):
        date = pd.to_datetime(df)
        return (date.month >= 8) & (date.month <= 10)
    def is_winter(df):
        date = pd.to_datetime(df)
        return (date.month >= 11) | (date.month <= 1)
    df_copy=pd.DataFrame(df_date.values,columns=['DATE'])
    df_copy['DATE']=pd.to_datetime(df_copy['DATE'])
    df_copy['day']=df_copy['DATE'].dt.day
    df_copy['month']=df_copy['DATE'].dt.month
    df_copy['year']=df_copy['DATE'].dt.year
    df_copy['week']=df_copy['DATE'].dt.week
    df_copy['dayofweek']=df_copy['DATE'].dt.dayofweek
    df_copy['weekend']=[1 if i>=5 else 0 for i in df_copy['DATE'].dt.weekday]
    df_copy['quarter']=df_copy['DATE'].dt.quarter
    df_copy['month_start']=df_copy['DATE'].dt.is_month_start.apply(lambda x : 1 if x==True else 0)
    df_copy['month_end']=df_copy['DATE'].dt.is_month_end.apply(lambda x : 1 if x==True else 0)
    df_copy['leap_year']=df_copy['DATE'].dt.is_leap_year.apply(lambda x : 1 if x==True else 0)
    df_copy['Q-MAR'] = df_copy['DATE'].dt.to_period('Q-MAR')
    df_copy['fiscal year'] = df_copy['Q-MAR'].dt.qyear
    df_copy['Q-MAR'] = df_copy['DATE'].dt.to_period('Q-MAR').astype("object")
    df_copy['ymd'] = (df_copy['year']*7)*10000+df_copy['month']*12*100+df_copy['day']*30
    df_copy['sin'] = np.sin(2 * np.pi * df_copy['ymd']/365)
    df_copy['cos'] = np.cos(2 * np.pi * df_copy['ymd']/365)
    df_copy['is_spring'] = df_copy['DATE'].apply(is_spring).astype(int)
    df_copy['is_summer'] = df_copy['DATE'].apply(is_summer).astype(int)
    df_copy['is_autumn'] = df_copy['DATE'].apply(is_autumn).astype(int)
    df_copy['is_winter'] = df_copy['DATE'].apply(is_winter).astype(int)

    return df_copy
def datetime_ohe(df_date):
    """
    All these transformed values transform to like OneHotEncoder type all features. All unique values is new features in the DataFrame. 

    Parameters
    ----------
    df_date : DataFrame / Series TYPE
        Datetime DataFrame/Series.

    Returns
    -------
    TYPE

    """
    def is_spring(df):
        date = pd.to_datetime(df)
        return (date.month >= 2) & (date.month <= 4)
    def is_summer(df):
        date = pd.to_datetime(df)
        return (date.month >= 5) & (date.month <= 7)
    def is_autumn(df):
        date = pd.to_datetime(df)
        return (date.month >= 8) & (date.month <= 10)
    def is_winter(df):
        date = pd.to_datetime(df)
        return (date.month >= 11) | (date.month <= 1)
    clmns=['day','month', 'dayofweek', 'hour']
    other_cols = [f'day_{i}' for i in range(1,32)]+[f'dayofweek{i}' for i in range(1, 8)] + [f'hour_{i}' for i in range(1, 25)] + [f'month_{i}' for i in range(1, 13)]

    df_copy=pd.DataFrame(df_date.values,columns=['DATE'])
    df_copy['DATE']=pd.to_datetime(df_copy['DATE'])
    df_copy['day']=df_copy['DATE'].dt.day
    df_copy['month']=df_copy['DATE'].dt.month
    df_copy['year']=df_copy['DATE'].dt.year
    df_copy['week']=df_copy['DATE'].dt.week
    df_copy['dayofweek']=df_copy['DATE'].dt.dayofweek
    df_copy['weekend']=[1 if i>=5 else 0 for i in df_copy['DATE'].dt.weekday]
    df_copy['quarter']=df_copy['DATE'].dt.quarter
    df_copy['month_start']=df_copy['DATE'].dt.is_month_start.apply(lambda x : 1 if x==True else 0)
    df_copy['month_end']=df_copy['DATE'].dt.is_month_end.apply(lambda x : 1 if x==True else 0)
    df_copy['leap_year']=df_copy['DATE'].dt.is_leap_year.apply(lambda x : 1 if x==True else 0)
    df_copy['Q-MAR'] = df_copy['DATE'].dt.to_period('Q-MAR')
    df_copy['fiscal year'] = df_copy['Q-MAR'].dt.qyear
    df_copy['Q-MAR'] = df_copy['DATE'].dt.to_period('Q-MAR').astype("object")
    df_copy['ymd'] = (df_copy['year']*7)*10000+df_copy['month']*12*100+df_copy['day']*30
    df_copy['sin'] = np.sin(2 * np.pi * df_copy['ymd']/365)
    df_copy['cos'] = np.cos(2 * np.pi * df_copy['ymd']/365)
    df_copy['is_spring'] = df_copy['DATE'].apply(is_spring).astype(int)
    df_copy['is_summer'] = df_copy['DATE'].apply(is_summer).astype(int)
    df_copy['is_autumn'] = df_copy['DATE'].apply(is_autumn).astype(int)
    df_copy['is_winter'] = df_copy['DATE'].apply(is_winter).astype(int)

    hour=df_copy['DATE'].dt.hour.value_counts()
    if hour[0]!=df_copy.shape[0]:
        df_copy['hour']=df_copy['DATE'].dt.hour
    else:
        other_cols = [f'day_{i}' for i in range(1,32)]+[f'dayofweek{i}' for i in range(1, 8)] + [f'month_{i}' for i in range(1, 13)]
        clmns.remove('hour')
    for column in clmns:
        index = [col for col in df_copy.columns if col!=column]
        df_copy = df_copy.pivot_table(index=index, columns=[column], aggfunc=np.count_nonzero).fillna(0).astype(bool).add_prefix(f'{column}_').reset_index()

    df_base = pd.DataFrame(columns= ['DATE'] + other_cols)
    df_copy = pd.concat([df_base, df_copy])
    df_copy[df_copy.columns[1:]] = df_copy[df_copy.columns[1:]].fillna(0).astype(int)

    df_copy['weekend']=[1 if i>=5 else 0 for i in df_copy['DATE'].dt.weekday]
    df_copy['is_spring'] = df_copy['DATE'].apply(is_spring).astype(int)
    df_copy['is_summer'] = df_copy['DATE'].apply(is_summer).astype(int)
    df_copy['is_autumn'] = df_copy['DATE'].apply(is_autumn).astype(int)
    df_copy['is_winter'] = df_copy['DATE'].apply(is_winter).astype(int)
    #df_copy['date']=df_copy['Tarih'].dt.date
    #df_copy['date']=pd.to_datetime(df_copy['date'])
    return df_copy