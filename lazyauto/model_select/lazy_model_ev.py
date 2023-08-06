from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import optuna

class model_select(TransformerMixin):
    def __init__(self,transform_pipeline,X_test,y_test,n_trial=20,optuna_param=None,optuna_direction='minimize',classifier=False,estimator=False):
        """
        Parameters
        ----------
        transform_pipeline :sklearn.pipeline.Pipeline TYPE
            Pipeline module without model steps.
        X_test : Series TYPE
            X_test data.
        y_test : Series TYPE
            y_test data.
        n_trial : Int TYPE, optional
            n_trial variable for Optuna. The default is 20.
        optuna_param : dict, optional
            Hyperparamaters dict for optuna.If you don't give value,the parameter search function will run first. The default is None.
        optuna_direction : String, optional
            Minimize or Maximize for Optuna model Score. The default is 'minimize'.
        classifier : String, optional
            The model problem you want to establish is Regression or Classifier. The default is False.
        estimator : Object, optional
            If you specify an external estimator, LazyEstimator will not work. The default is False.
        """
        
        self.classifier=classifier
        self.pipe=transform_pipeline
        self.X_test=X_test
        self.y_test=y_test
        self.optuna_param=optuna_param
        self.optuna_direction=optuna_direction
        self.estimator=estimator
        self.n_trial=n_trial
    def prt(self,title=''):
        """
        This function using to console printing.

        Parameters
        ----------
        title : String TYPE, optional
             The default is ''.
        """
        print("=" * 2*len(title))
        c_title=title.center(2*len(title))
        print(c_title)
        print("=" * 2*len(title))
    def pipe_fit(self,X,y):
        """
        For the defined pipeline, it transforms X_test appropriately through the fit pipeline and prepares it for optuna.    

        Parameters
        ----------
        X : Series,DataFrame TYPE
            X_train.
        y : Series,DataFrame
            y_train (target).

        Returns
        -------
        X_train_trans : Series TYPE
            Transformed X_train.
        X_test_trans :Series TYPE
            Transformed X_test.

        """
        from sklearn import set_config            
        self.prt('Pipe Transformation Started... ')
        set_config(display="diagram")
        pipe_transform=self.pipe
        X_train_trans=pipe_transform.fit_transform(X,y)
        X_test_trans=pipe_transform.transform(self.X_test)
        return X_train_trans,X_test_trans
    class find_params:
        def __init__(self,est):
            """
            It is intended for finding parameters for an estimator found or defined by Lazyestimator and for which no parameter is defined.
            Parameters
            ----------
            est : Object TYPE
                Estimator.

            """
            self.est=est
            self.ridge_params={}
            self.lasso_params={}
            self.elasticnet_params={}
            self.dt_params={}
            self.randomforest_params={}
            self.gradient_params={}
            self.xgb_params={}
            self.lgbm_params={}
            self.catboost_params={}
            self.reg_result=self.reg()
            self.clf_result=self.clf()
        def reg(self):
            """
            This Function define to parameters for Regression Estimators.
            
            """
            
            "Regressor Models"
            from sklearn.linear_model import Ridge 
            from sklearn.linear_model import Lasso 
            from sklearn.linear_model import ElasticNet 
            from sklearn.tree import DecisionTreeRegressor 
            from sklearn.ensemble import RandomForestRegressor 
            from sklearn.ensemble import GradientBoostingRegressor 
            from xgboost import XGBRegressor 
            from lightgbm import LGBMRegressor 
            from sklearn.svm import SVR 
            from catboost import CatBoostRegressor 
            if self.est == Ridge:
                self.ridge_params={'alpha': [0.1, 1.0, 10.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr']}
                return self.ridge_params
            elif self.est == Lasso:
                self.lasso_params={'alpha': [0.1, 1.0, 10.0], 'max_iter': [1000, 2000, 5000], 'selection': ['cyclic', 'random']}
                return self.lasso_params
            elif self.est == ElasticNet:
                self.elasticnet_params={'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.25, 0.5, 0.75], 'max_iter': [1000, 2000, 5000]}
                return self.elasticnet_params
            elif self.est == DecisionTreeRegressor:
                self.dt_params={'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 3]}
                return self.dt_params
            elif self.est == RandomForestRegressor:
                self.randomforest_params={'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}
                return self.randomforest_params
            elif self.est == GradientBoostingRegressor:
                self.gradient_params={'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                return self.gradient_params
            elif self.est == XGBRegressor:
                self.xgb_params={'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                return self.xgb_params
            elif self.est == LGBMRegressor:
                self.lgbm_params={'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                return self.lgbm_params
            elif self.est == CatBoostRegressor:
                self.catboost_params={'iterations': [100, 200, 500],'learning_rate': [0.01, 0.1, 0.2],'depth': [4, 6, 8],'l2_leaf_reg': [1, 3, 5],
                        'border_count': [32, 64, 128]}
                return self.catboost_params
            elif self.est == SVR:
                self.svr_params={'C': [0.1, 1.0, 10.0], 'epsilon': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
                return self.svr_params
            return []
        def clf(self):
            """
            This Function define to parameters for Regression Estimators.
   
            """
            "Classification Models"
            from sklearn.linear_model import LogisticRegression 
            from sklearn.neighbors import KNeighborsClassifier 
            from sklearn.svm import SVC 
            from sklearn.tree import DecisionTreeClassifier 
            from sklearn.ensemble import RandomForestClassifier 
            from sklearn.ensemble import GradientBoostingClassifier 
            from xgboost import XGBClassifier 
            from lightgbm import LGBMClassifier 
            from catboost import CatBoostClassifier 
            from sklearn.naive_bayes import GaussianNB
            if self.est == LogisticRegression:
                self.logreg_params={'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs'], 'max_iter': [100, 200, 500]}
                return self.logreg_params
            elif self.est == KNeighborsClassifier:
                self.knn_params={'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance'], 'p': [1, 2, 3]}
                return self.knn_params
            elif self.est == SVC:
                self.svc_params={'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'decision_function_shape': ['ovo', 'ovr']}
                return self.svc_params
            elif self.est == DecisionTreeClassifier:
                self.dt_params={'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
                return self.dt_params
            elif self.est == RandomForestClassifier:
                self.randomforest_params={'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
                return self.randomforest_params
            elif self.est == GradientBoostingClassifier:
                self.gradient_params={'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                return self.gradient_params
            elif self.est == XGBClassifier:
                self.xgb_params={'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                return self.xgb_params
            elif self.est == LGBMClassifier:
                self.lgbm_params={'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                return self.lgbm_params
            elif self.est == CatBoostClassifier:
                self.catboost_params={'iterations': [100, 200, 500],'learning_rate': [0.01, 0.1, 0.2],'depth': [4, 6, 8],'l2_leaf_reg': [1, 3, 5],
                        'border_count': [32, 64, 128]}
                return self.catboost_params
            elif self.est == GaussianNB:
                self.svr_params={}
                return self.svr_params
            return []
        def convert_params(self,param_dict):
            """
            Converts the found parameters into a dictionary of parameters suitable for the Optuna (Trial) object.

            Parameters
            ----------
            param_dict : Dict TYPE
                Paramter dict basicly not have a trial object.

            Returns
            -------
            Dict TYPE
                Returns a new dict according to the Trial object.

            """
            def o_params(trial):
                params_c={}
                for key,value in param_dict.items():
                    params_c[key]=trial.suggest_categorical(key,value)
                return params_c
            return o_params

    def lazy_model(self,X,y):
        """
        Returns the estimator that gives the best score among the estimators defined in the list.


        Parameters
        ----------
        X : Series / DataFrame TYPE
            DESCRIPTION.
        y : Series / DataFrame TYPE
            DESCRIPTION.

        Returns
        -------
        Object TYPE
            Estimator Returned.

        """
        
        "Regressor Models"
        from sklearn.linear_model import Ridge 
        from sklearn.linear_model import Lasso 
        from sklearn.linear_model import ElasticNet 
        from sklearn.tree import DecisionTreeRegressor 
        from sklearn.ensemble import RandomForestRegressor 
        from sklearn.ensemble import GradientBoostingRegressor 
        from xgboost import XGBRegressor 
        from lightgbm import LGBMRegressor 
        from sklearn.svm import SVR 
        from catboost import CatBoostRegressor 
        "Classification Models"
        from sklearn.linear_model import LogisticRegression 
        from sklearn.neighbors import KNeighborsClassifier 
        from sklearn.svm import SVC 
        from sklearn.tree import DecisionTreeClassifier 
        from sklearn.ensemble import RandomForestClassifier 
        from sklearn.ensemble import GradientBoostingClassifier 
        from xgboost import XGBClassifier 
        from lightgbm import LGBMClassifier 
        from catboost import CatBoostClassifier 
        from sklearn.naive_bayes import GaussianNB

        from sklearn.metrics import f1_score,mean_squared_error

        verbose_list=[GradientBoostingRegressor,
                            XGBRegressor,
                            LGBMRegressor,
                            CatBoostRegressor,
                            GradientBoostingClassifier,
                            XGBClassifier,
                            LGBMClassifier,
                            CatBoostClassifier]

        regressor=[Ridge,Lasso,ElasticNet,DecisionTreeRegressor,RandomForestRegressor,GradientBoostingRegressor,
                  XGBRegressor,LGBMRegressor,SVR,CatBoostRegressor]
        classifications=[LogisticRegression,KNeighborsClassifier,SVC,DecisionTreeClassifier,RandomForestClassifier,
                        GradientBoostingClassifier,XGBClassifier,LGBMClassifier,GaussianNB,CatBoostClassifier]
        results=[]
        if self.estimator==False:
            self.prt("Lazy Best Estimator Searching Started...")
            if self.classifier==True:
                for clf in classifications:
                    model=clf()
                    model.fit(self.X_train_trans,y)
                    y_pred=model.predict(self.X_test_trans)
                    f1 = f1_score(self.y_test, y_pred, average='weighted')
                    results.append(f1)
                results=np.array(results)
                max_index = np.argmax(results)
                self.prt('Best Estimator')
                print(classifications[max_index])
                self.prt('              ')
                return classifications[max_index]
            else:
                for est in regressor:
                    model=est()
                    model.fit(self.X_train_trans,y)
                    y_pred=model.predict(self.X_test_trans)
                    mse=mean_squared_error(self.y_test,y_pred)
                    results.append(mse)
                results=np.array(results)
                min_index = np.argmin(results)
                self.prt('Best Estimator')
                print(regressor[min_index])
                return regressor[min_index],verbose_list
    
    def fit(self,X,y):
        from functools import partial
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error,classification_report,f1_score

        def objective(trial,best_est,X_train,y_train,X_test,y_test,param,verbose):
            def convert_object_param(trial,model):
                """
                It translates the parameter values to " max_depth:trial.suggest_categorical" and maps the parameters to the model.

                Parameters
                ----------
                trial : TYPE
                    DESCRIPTION.
                model : TYPE
                    DESCRIPTION.

                Returns
                -------
                TYPE
                    DESCRIPTION.

                """
                
                params={}
                for key,value in param(trial).items():
                    params[key]=value
                return model(**params)

            model=convert_object_param(trial,best_est) # Returns the model with parameters mapped together with the best estimator.
            if (model=='xgboost.sklearn.XGBRegressor') | (model=='xgboost.sklearn.XGBClassifier'):
                eval_set = [(X_train, y_train), (X_test, y_test)]
                model.fit(X_train,y_train,eval_set=eval_set,early_stopping_rounds=10, verbose=0)
            else:
                model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            #avg_score=mean_squared_error(y_pred,y_test)
            # Classifier and regression model metrics are reserved for metrics.
            if self.classifier==True: # 
                f1 = f1_score(y_test, y_pred, average='weighted')
                report = classification_report(y_test, y_pred)
                print(report)
                return f1
            else:
                scores = cross_val_score(model, X_train, y_train, cv=5,scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                return -avg_score
        
        self.X_train_trans,self.X_test_trans=self.pipe_fit(X,y) # Pipeline fit and transform Starts
        best_est,v_list=self.lazy_model(self.X_train_trans,y) # Lazyestimator starts
        #If parameters not defined , call the before defined finds parameters
        if self.optuna_param==None: 
            if self.find_params(best_est).reg_result !=[]:
                param=self.find_params(best_est).convert_params(self.find_params(best_est).reg_result)
            else:
                param=self.find_params(best_est).convert_params(self.find_params(best_est).clf_result)

        else:
            param=self.optuna_param
            
        self.prt('Optuna Has Started...')
        study = optuna.create_study(direction=self.optuna_direction) 
        objective_with_param=partial(objective,best_est=best_est,X_train=self.X_train_trans,y_train=y,X_test=self.X_test_trans,
                                     y_test=self.y_test,param=param,verbose=v_list)

        
        study.optimize(objective_with_param, n_trials=self.n_trial) # Optuna start
        best_params = study.best_params
        self.prt('Best Params')
        print(best_params) #Print best params
        self.model=best_est(**best_params)
        self.model.fit(self.X_train_trans,y)
        
        return self
    def predict(self):
        """
        Predicts X_test_trans previously transformed from the pipeline and returns predict.

        Returns
        -------
        pred :Array TYPE

        """
        pred=self.model.predict(self.X_test_trans)
        return pred