from lazyauto.Eda.eda import dedect_features,skewness,null_val
from sklearn import datasets
import pandas as pd

diab=datasets.load_diabetes(as_frame=True)
df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])

def test_dedect_features():
    date_cols,num_but_cat,cat_col,cat_but_num,num_col=dedect_features(df)
    assert (date_cols,num_but_cat,cat_col,cat_but_num,num_col) == ([],['sex'],[],[],['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target'])
def test_skewness():
    assert skewness(df) == ['bmi','s3','s4']
def test_null_val():
    assert any(null_val(df)[0]==0)