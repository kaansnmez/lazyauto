def drop_outliers(df,drop_index):
    """
    It drops the index list resulting from the outlier analysis from the dataframe.   

    Parameters
    ----------
    df :DataFrame TYPE
        
    drop_index : Dataframe TYPE
        DESCRIPTION.

    Returns
    -------
    df_out : TYPE
        DESCRIPTION.

    """
    f=df.shape[0]
    df_out=df.copy()
    print("Shape: ",f)
    for i in drop_index.index.to_list():
        try:
            df_out.drop(i,axis=0,inplace=True)
        except KeyError:
            continue
    df_out=df_out.reset_index(drop=True)
    l=df_out.shape[0]
    print("Later Dropped Outliers Shape: ",l)
    print("Result: ",f-l)
    print('Completed..')
    return df_out