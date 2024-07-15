########
lazyauto
########


This library have some Eda graphics and Time features easy transform functions and model development.

.. image:: https://img.shields.io/pypi/v/lazyauto.svg
        :target: https://pypi.python.org/pypi/lazyauto

.. image:: https://img.shields.io/travis/kaansnmez/lazyauto.svg
        :target: https://travis-ci.com/kaansnmez/lazyauto

.. image:: https://readthedocs.org/projects/lazyauto/badge/?version=latest
        :target: https://lazyauto.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Installation
************

Available as a `PyPI <https://pypi.python.org/pypi/lazyauto>`_ package:

.. code-block:: bash

    pip install lazyauto

Usage
*********
EDA
======
This library have a lot of function and some functions have html based looking great for  using on JupiterNotebook.
All functions is listed below.

========    
show
========
    

    Helper function to set dataframe presentation style Html Based like ``JupiterNotebook``.
    
    .. code-block:: python
    
        from lazyauto.Eda import eda
        from lazyauto.Eda.eda import describes
        
        display(eda.show(describes(train_df),'Train:Describe'))
    
    Result is: 
    
    .. image:: doc_image/show.png
    
===========    
skewness
===========
    
    This function evaluates whether the features in the given dataframe are skewed and returns a list.
    
    .. code-block:: python
    
        from sklearn import datasets
        from lazyauto.Eda.eda import skewness
        
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
        skw=skewness(df)
    
    Result:
    
    .. code-block:: python
    
        >>> skw 
        ['bmi', 's3', 's4']
        
=========    
null_val
=========

    This function returns a list of null values as Dataframe.
    Because easy to use with ``show`` function in ``JupyterNotebook``
    
    .. code-block:: python
    
        from sklearn import datasets
        from lazyauto.Eda.eda import null_val
        
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
        na=null_val(df)
    
    Result:
    
    .. code-block:: python
    
        >>> na 
                0
        age     0
        sex     0
        bmi     0
        bp      0
        s1      0
        s2      0
        s3      0
        s4      0
        s5      0
        s6      0
        target  0

================    
describe_object
================    

    This function returns the describe function for features other than continues variable.
    
    .. code-block:: python
    
        from lazyauto.Eda import eda
        from lazyauto.Eda.eda import describes_object
        display(eda.show(describes_object(train_df),'Train:Describe'))
    
    Result:
    
    .. image:: doc_image/describe_object.PNG

==============    
describes
==============

    This function returns the describe function for features other than categoric variable.    
    
    .. code-block:: python
    
        from lazyauto.Eda import eda
        from lazyauto.Eda.eda import describes
        display(eda.show(describes(train_df),'Train:Describe'))
    
    Result:
    
    .. image:: doc_image/describe.PNG

==============    
unique_val
==============

    This function finds unique data that is different in categoric variables between two dataframes. 
    It returns data that is in dataframe 2 but not in dataframe 1.
    This function return 2 values. One is dataframe for display on JupyterNotebook ,Second is dict.
    
    .. code-block:: python
        from lazyauto.Eda.eda import show
        from lazyauto.Eda.eda import unique_val
        
        diff=unique_val(train_df,test_df,['V_10','SOT'])
        display(show(diff[0],'Test non-data Train Df'))
        print(diff[1])
    
    Result:
    
    .. image:: doc_image/unique_val.PNG
    
===============
dedect_features
===============

    This function separates data types. And returns them. It takes 3 variables. The first is dataframe, the second is unique val threshold limit for numeric data.
    The third is unique val threshold limit for Categoric variables. 
    These are for extracting features that are numeric but behave like categoric or like categoric but behave like numeric.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import dedect_features
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
        
        date_cols,num_but_cat,cat_col,cat_but_num,num_col=dedect_features(df,show=True,20,30)
    
    Result:
    
    .. code-block:: python
    
        >>>
        Date Columns:  []
        Numeric But Categoric Columns:  ['sex']
        Categoric Columns:  []
        Categoric But Numeric Columns:  []
        Numeric Columns:  ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target']
==========    
corr
==========

    Plots the edited correlation heatmap.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import corr
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
        
        corr(df)
    
    Result:
    
    .. image:: doc_image/corr.PNG
    
==========    
cat_plot
========== 

    Draws distributions of categorical data.It can produce more than one figure output according to the number of categorical variables.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import cat_plot
        
        csv=pd.read_csv("data\\House_Rent_Dataset.csv")

        cat_plot(df,target='target','Distributions')
     
    Result:
     
    .. image:: doc_image/cat_plot.PNG
    .. image:: doc_image/cat_plot_1.png
    .. image:: doc_image/cat_plot_2.png
    
==========    
pairplot
==========    
    Draws Boxplot and Scatterplot distributions of Continues values. Plotly is used. Opens automatically in the default browser on local as well.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import pairplot
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
         
        pairplot(df,'Distributions')
     
    Result:
     
    .. image:: doc_image/pairplot.PNG
    
    
==================    
outlier_dedection
==================

    Isolation forest is used. For the best comparison based on score, the estimator must be specified. 
    If the Contaminations value is not entered, it returns the Scores of all values by default. 
    In this way, the user can see the scores of all default defined values, make a selection and run it again. 
    If graph drawing is not desired, ``graph`` can be set to False.
    If True, PCA is applied and 2-dimensional residual data is marked and plotted on the basis of features. 
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import outlier_dedection
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
         
        outlier_dedection(df,target='target',model=LGBMRegressor(),contaminations=[0.1],graph=True)
     
    Result:
     
    .. image:: doc_image/isolation_pca.PNG
    
    .. image:: doc_image/isolation_ds.PNG

============    
preperation
============

Includes pre-processing functions.

=============
drop_outlier
=============
    Drops the object returned from eda.outlier_dedection over its indexes.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import outlier_dedection
        from lazyauto.preperation.drop_outlier import drop_outliers
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
        
        outlier=outlier_dedection(df,target='target',model=LGBMRegressor(),contaminations=[0.1],graph=True)
        dropped_df=drop_outliers(df,outlier)
     
    Results:   
    
    .. code-block:: python
     
        >>>
        Shape:  442
        Later Dropped Outliers Shape:  397
        Result:  45
        Completed.. 
        >>> dropped_df.shape
        (397,11)
time_transform
======
Includes feature extraction functions for datetime features.

================
datetime_simple
================
    
    This function create a new features like day,month,year,week ..etc and returning dataframe. 
     
    .. code-block:: python
    
        from lazyauto.time_transform.date_time_transform import datetime_simple
        from lazyauto.Eda.eda import null_vall
        train=pd.read_csv("data/train.csv")
        train['date']=pd.to_datetime(train['date'])
        train_date=date_time_transform.datetime_simple(train['date'])
        
    .. code-block:: python
    
        >>>train_date.columns
        
        Index(['DATE', 'day', 'month', 'year', 'week', 'dayofweek', 'weekend',
               'quarter', 'month_start', 'month_end', 'leap_year', 'Q-MAR',
               'fiscal year', 'ymd', 'sin', 'cos', 'is_spring', 'is_summer',
               'is_autumn', 'is_winter'],
              dtype='object')
        >>>null_val(train_date)
        
                     0
        DATE         0
        day          0
        month        0
        year         0
        week         0
        dayofweek    0
        weekend      0
        quarter      0
        month_start  0
        month_end    0
        leap_year    0
        Q-MAR        0
        fiscal year  0
        ymd          0
        sin          0
        cos          0
        is_spring    0
        is_summer    0
        is_autumn    0
        is_winter    0

=============        
datetime_ohe
=============

    This function transforms datetime data into day,month,year etc. with OHE approach.
        
    .. code-block:: python
    
        from lazyauto.time_transform.date_time_transform import datetime_ohe
        from lazyauto.Eda.eda import null_vall
        train=pd.read_csv("data/train.csv")
        train['date']=pd.to_datetime(train['date'])
        train_date=date_time_transform.datetime_simple(train['date'])
        
    .. code-block:: python
    
        >>>train_date.columns
        Index(['DATE', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7',
             'day_8', 'day_9', 'day_10', 'day_11', 'day_12', 'day_13', 'day_14',
             'day_15', 'day_16', 'day_17', 'day_18', 'day_19', 'day_20', 'day_21',
             'day_22', 'day_23', 'day_24', 'day_25', 'day_26', 'day_27', 'day_28',
             'day_29', 'day_30', 'day_31', 'dayofweek1', 'dayofweek2', 'dayofweek3',
             'dayofweek4', 'dayofweek5', 'dayofweek6', 'dayofweek7', 'month_1',
             'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
             'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'year',
             'week', 'weekend', 'quarter', 'month_start', 'month_end', 'leap_year',
             'Q-MAR', 'fiscal year', 'ymd', 'sin', 'cos', 'is_spring', 'is_summer',
             'is_autumn', 'is_winter', 'dayofweek_0', 'dayofweek_1', 'dayofweek_2',
             'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6'],
            dtype='object')
        >>>null_val(train_date)
                     0
        DATE         0
        day_1        0
        day_2        0
        day_3        0
        day_4        0
                ..
        dayofweek_2  0
        dayofweek_3  0
        dayofweek_4  0
        dayofweek_5  0
        dayofweek_6  0

        [74 rows x 1 columns]

lazy_model_ev 
===============
This class has multiple functions and all functions are interconnected. 
Pipeline performs the final step of model development and hyperparameter optimization for a model that has been built and all preprocessing and extraction parts have been completed. 
If an estimator is not selected, it selects its best estimator (min score or max score) and automatically runs and transforms the pipeline object for X_test and X_train. 
After model selection, it uses Optuna for Hyperparameter Opt, and if parameters are defined, it can import them from outside or optimize within the parameters defined in itself. 

.. code-block:: python

    from lazyauto.model_select.lazy_model_ev import model_select
    from lazyauto.Eda import eda
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    
    diab=datasets.load_diabetes(as_frame=True)
    df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
    
    date_cols,num_but_cat,cat_col,cat_but_num,num_col=eda.dedect_features(df,show=True)
    ord_col=date_cols+num_but_cat+cat_but_num
    num_col.remove('target')
    X=df.drop(['target'],axis=1)
    y=df['target']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y, \
        test_size=0.2, random_state=0)
        
    from sklearn.preprocessing import MinMaxScaler,RobustScaler,OrdinalEncoder,StandardScaler,OneHotEncoder,FunctionTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    
    cardinal_transformer=Pipeline(steps=[
        ('N_encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    numeric_transformer=Pipeline(steps=[
        ('scaler_robust',StandardScaler())
    ])
    categoric_transformer=Pipeline(steps=[
        ('OHE',OneHotEncoder(drop='first',handle_unknown='ignore'))
    ])
    func_trans=Pipeline(steps=[
        ('FT',FunctionTransformer(lambda x : np.where(x < 0, 0, np.log1p(x))))
    ])
    preprocessing_transform=ColumnTransformer(transformers=[
        ('Numeric_trans',numeric_transformer,num_col),
        ('card_trans',cardinal_transformer,ord_col),
    ])
    pipe=Pipeline([
        ('column_transformers',preprocessing_transform)
    ])
      
    ml=model_select(transform_pipeline=pipe,X_test=X_test,y_test=y_test,n_trial=5,optuna_direction='minimize',classifier=False)
    ml.fit(X_train,y_train)
    
.. code-block:: python

    >>> 
    ============================
           Best Estimator       
    ============================
    <class 'sklearn.linear_model._coordinate_descent.ElasticNet'>
    ==========================================
              Optuna Has Started...           
    ==========================================
    [I 2023-08-06 18:34:54,092] Trial 1 finished with value: 4790.025069386211 and parameters: {'alpha': 10.0, 'l1_ratio': 0.25, 'max_iter': 2000}. Best is trial 0 with value: 2967.2739952009483.
    [I 2023-08-06 18:34:54,097] Trial 2 finished with value: 2874.8298628850143 and parameters: {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 5000}. Best is trial 2 with value: 2874.8298628850143.
    [I 2023-08-06 18:34:54,101] Trial 3 finished with value: 4790.025069386211 and parameters: {'alpha': 10.0, 'l1_ratio': 0.25, 'max_iter': 1000}. Best is trial 2 with value: 2874.8298628850143.
    [I 2023-08-06 18:34:54,105] Trial 4 finished with value: 4122.589373510469 and parameters: {'alpha': 10.0, 'l1_ratio': 0.75, 'max_iter': 5000}. Best is trial 2 with value: 2874.8298628850143.
    ======================
         Best Params      
    ======================
    {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 5000}
    
We see that it is ElasticNet. We define manual parameters and run it again.
 
.. code-block:: python

    def optna_params(trial):
        params={
        'alpha' : trial.suggest_categorical('alpha', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]),
        'l1_ratio' : trial.suggest_discrete_uniform('l1_ratio', 0.1, 1.0, 0.01)
        }
        return params
        
    ml=model_select(transform_pipeline=pipe,X_test=X_test,y_test=y_test,n_trial=5,optuna_param=optna_params,optuna_direction='minimize',classifier=False)
    ml.fit(X_train,y_train)

.. code-block:: python
    
    >>>
    [I 2023-08-06 18:37:34,901] Trial 0 finished with value: 2882.365186812675 and parameters: {'alpha': 0.0, 'l1_ratio': 0.26}. Best is trial 0 with value: 2882.365186812675.
    [I 2023-08-06 18:37:34,908] Trial 1 finished with value: 2877.98529472333 and parameters: {'alpha': 0.01, 'l1_ratio': 0.71}. Best is trial 1 with value: 2877.98529472333.
    [I 2023-08-06 18:37:34,912] Trial 2 finished with value: 6048.509205527287 and parameters: {'alpha': 100.0, 'l1_ratio': 0.2}. Best is trial 1 with value: 2877.98529472333.
    [I 2023-08-06 18:37:34,916] Trial 3 finished with value: 3995.4323739370934 and parameters: {'alpha': 10.0, 'l1_ratio': 0.8}. Best is trial 1 with value: 2877.98529472333.
    [I 2023-08-06 18:37:34,927] Trial 4 finished with value: 2882.1837192613707 and parameters: {'alpha': 0.0001, 'l1_ratio': 0.18}. Best is trial 1 with value: 2877.98529472333.
    ======================
         Best Params      
    ======================
    {'alpha': 0.01, 'l1_ratio': 0.71}
    
    >>> ml.predict()
    array([237.91909772, 248.74425901, 164.0124327 , 120.51011937,
           186.45496246, 258.8468808 , 113.37605946, 188.02498302,
           151.13242029, 235.02386472, 170.94179689, 178.32381054,
           109.47711419,  92.71225284, 242.41537139,  88.91875579,
           154.75763539,  67.45635784, 101.25601208, 217.32561696,
           196.67317872, 160.29684431, 161.17698657, 157.39225171,
           197.71927976, 167.13819902, 118.97049015,  85.02764127,
           190.30977511, 160.14547978, 174.44455817,  85.15292631,
           145.72352411, 145.21542225, 140.9604623 , 195.93414085,
           165.20997213, 189.06423931, 129.03438721, 205.66189789,
            84.46691801, 163.58819311, 143.98154934, 183.79699208,
           177.3757117 ,  74.92607277, 141.7816472 , 139.11357299,
           120.60702682, 233.63905114, 161.3889688 ,  76.00628287,
           155.5166513 , 156.10410183, 236.67493798, 172.95795035,
           189.90149751, 119.12465929, 132.17563385, 168.82329251,
           213.70964089, 170.75921558, 158.05577804, 109.38002908,
           259.16702187, 151.86694055,  82.90954234, 229.93155699,
           201.76404651,  45.61729938,  79.71105015, 129.30588666,
           104.06781006, 144.24421601, 132.81697591, 188.59758212,
            98.32272063, 197.55707951, 218.77521522, 186.21091501,
           149.13202371, 208.03587707,  47.09948986, 205.93341284,
            76.4104104 ,  94.91521826, 144.99916429, 192.80180453,
           132.46338124])
