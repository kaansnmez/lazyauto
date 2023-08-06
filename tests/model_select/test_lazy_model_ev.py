import numpy as np
import pandas as pd
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

    
    
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder,FunctionTransformer
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

def optna_params(trial):
    params={
    'alpha' : trial.suggest_categorical('alpha', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]),
    'l1_ratio' : trial.suggest_discrete_uniform('l1_ratio', 0.1, 1.0, 0.01)
    }
    return params
def test_lazy_model_ev():
    ml=model_select(transform_pipeline=pipe,X_test=X_test,y_test=y_test,n_trial=5,optuna_param=optna_params,optuna_direction='minimize',classifier=False)
    ml.fit(X_train,y_train)
    pred=ml.predict()
    assert pred != []