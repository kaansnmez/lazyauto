from lazyauto.preperation.drop_outlier import drop_outliers
from lazyauto.Eda.eda import outlier_dedection
from sklearn import datasets
from lightgbm import LGBMRegressor
import pandas as pd

diab=datasets.load_diabetes(as_frame=True)
df=pd.DataFrame(diab['frame'],columns=diab['feature_names']+['target'])


def test_drop_outlier():
    outlier_index=outlier_dedection(df, target='target', model=LGBMRegressor(),contaminations=[0.1])
    before_drop=df.shape[0]
    dropped_df=drop_outliers(df,outlier_index).shape[0]
    assert before_drop>dropped_df
    