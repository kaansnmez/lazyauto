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

    * show
    
    Helper function to set dataframe presentation style Html Based like ``JupiterNotebook``.
    
    .. code-block:: python
    
        from lazyauto.Eda import eda
        from lazyauto.Eda.eda import describes
        
        display(eda.show(describes(train_df),'Train:Describe'))
    
    Result is: 
    
    .. image:: doc_image/show.png
    
    
    * skewness
    
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
        
    
    * null_val
    
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
    
    * describe_object
    
    This function returns the describe function for features other than continues variable.
    
    .. code-block:: python
    
        from lazyauto.Eda import eda
        from lazyauto.Eda.eda import describes_object
        display(eda.show(describes_object(train_df),'Train:Describe'))
    
    Result:
    
    .. image:: doc_image/describe_object.PNG
    
    * describes
    
    This function returns the describe function for features other than categoric variable.    
    
    .. code-block:: python
    
        from lazyauto.Eda import eda
        from lazyauto.Eda.eda import describes
        display(eda.show(describes(train_df),'Train:Describe'))
    
    Result:
    
    .. image:: doc_image/describe.PNG
    
    * unique_val
    
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
    

    * dedect_features
    
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
    
    * corr
    
    Plots the edited correlation heatmap.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import corr
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
        
        corr(df)
    
    Result:
    
    .. image:: doc_image/corr.PNG
    
    
    * cat_plot
    
    Draws Pie Chart and Barplot distributions of categorical data.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import cat_plot
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
         
        cat_plot(df,target='target','Distributions')
     
    Result:
     
    .. image:: doc_image/cat_plot.PNG
    
    
    * pairplot
    
    Draws Boxplot and Scatterplot distributions of Continues values. Plotly is used. Opens automatically in the default browser on local as well.
    
    .. code-block:: python
    
        from lazyauto.Eda.eda import pairplot
        from lightgbm import LGBMRegressor
        from sklearn import datasets
        diab=datasets.load_diabetes(as_frame=True)
        df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])
         
        pairplot(df,'Distributions')
     
    Result:
     
    .. image:: doc_image/pairplot.PNG
    
    
    
    * outlier_dedection
    
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
    




