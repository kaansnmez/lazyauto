import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import random
from time import sleep
import plotly.io as io
import warnings
warnings.filterwarnings('ignore')
io.templates.default = 'plotly_dark'
io.renderers.default = 'notebook_connected'


def show(df, caption=""):
    """Helper function to set dataframe presentation style Html Based like JupiterNotebook.""" # noqa:E501
    return df.style.background_gradient(cmap='Reds').set_caption(caption).set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'Red'),
            ('font-size', '18px'),
            ('font-weight', 'bold')
        ]}])


def skewness(df):
    """
    This function find skewness above 0.5 and return list. # noqa:E501

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    TYPE
        list.

    """
    skewness=df.skew()
    return skewness[skewness>0.5].index.to_list()


def null_val(df):
    """
    This function find if have null value. # noqa:E501

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    TYPE 
        Dataframe
    """
    return df.isna().sum().to_frame()


def describe_object(df):
    """
    This function call describe and just other than "number" and return. # noqa:E501

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    return df.describe(exclude='number').T
def describes(df):
    return df.describe(percentiles=[0.01,0.05, 0.1 , 0.25,  0.5,  0.75, 0.9 , 0.99]).T


def unique_val(df_train,df_test,cat_col):
    """
    

    Parameters
    ----------
    df_train : DataFrame
        Train Data.
    df_test : DataFrame
        Test Data.
    cat_col : list
        Catagoric columns in data.

    Returns
    -------
    v_df : TYPE
        DESCRIPTION.
    compare : TYPE
        DESCRIPTION.

    """
    compare={}
    v_df=pd.DataFrame()
    for ind,col in enumerate(cat_col):  
        a=list(df_train[col].unique())
        b=list(df_test[col].unique())
        res = [x for x in b if x not in a]
        compare[col]=res
        v_df.loc[0,col]=len(res)
    return v_df,compare


def dedect_features(df,show=False,nc_th=20,cn_th=30):
    '''
    This function dedect features type and return as a list. # noqa:E501
    
        Parameters
        ----------
        df : DataFrame
            Main Df
        nc_th : Integer, 
            Numeric but categorical threshold limit. The default is 20.
        cn_th : Integer, 
            Categorical but numeric threshold limit. The default is 30.

        Returns
        -------
        date_cols : List
            Date Types Columns List.
        num_but_cat : List
            Numeric but Categorical Types Columns List.
        cat_col : List
            Categorical Types Columns List.
        cat_but_num : List
            Categorical but Numeric Columns List.
        num_col : List
            Date Types Columns List.

    '''
    date_cols=[col for col in df if df[col].dtypes==("datetime64[ns]")]

    num_but_cat=[col for col in df.select_dtypes(['float','integer']).columns if df[col].value_counts().shape[0]<nc_th]
    cat_col=df.select_dtypes(['object','category']).columns.to_list()
    cat_but_num=[col for col in df.select_dtypes(['object','category']).columns if df[col].value_counts().shape[0]>cn_th]
    cat_col=[col for col in cat_col if col not in cat_but_num]

    num_col=df.select_dtypes(['integer','float']).columns.to_list()
    num_col=[col for col in num_col if col not in num_but_cat]
    if show==True:
        print("Date Columns: ",date_cols)
        print("Numeric But Categoric Columns: ",num_but_cat)
        print("Categoric Columns: ",cat_col)
        print("Categoric But Numeric Columns: ",cat_but_num)
        print("Numeric Columns: ",num_col)
    
    return date_cols,num_but_cat,cat_col,cat_but_num,num_col
    

def corr(df,title):
    """
    Parameters
    ----------
    df : DataFrame
        
    title : String
        Heatmap Title.
    """
    
    corrr=df.astype(float).corr()
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(15, 15))
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=20)
    sns.heatmap(corrr, linewidths=0.1, vmax=1.0, vmin=-1.0,mask=np.triu(np.ones_like(corrr),k=1), 
        square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 11, "weight": "bold"},fmt=".2f",
               cbar_kws={"shrink":0.8})


def cat_plot(df,target,title='',drop=[]):
    """
    Draws the distrubiton graph of categoric values ​​with piechart and bar plot.


    Parameters
    ----------
    df : DataFrame
        
    target : String
        Target Features in data.
    title : String, optional
        Subplot Title. The default is ''.
    drop : List, optional
        If you want drop any feature , you have to defined list. The default is [].# noqa:E501
        
    """
    date_cols,num_but_cat,cat_col,cat_but_num,num_col=dedect_features(df,show=False)
    sleep(0.1)
    cols=df[cat_col+num_but_cat].columns
    n_cols=2
    n_rows=len(cols)
    plt.rcParams['text.color'] = 'w'
    plt.rcParams['xtick.color'] = 'w'
    plt.rcParams['ytick.color'] = 'w'
    plt.rcParams['axes.labelcolor'] = 'w'
    plt.rcParams['axes.edgecolor'] = 'w'
    plt.rcParams['axes.titlesize']=20.0
    plt.rcParams['axes.labelsize']=10.0
    plt.rcParams['axes.titlepad']=10.0
    plt.rcParams['axes.titlecolor']='r'
    plt.rcParams['axes.titleweight']=12.0
    plt.rcParams['font.size']=11.0
    sleep(1)
    
    graph_max=3
    g_height=2
    len_col=len(cols.to_list())
    loop_count=int(len_col/graph_max)
    leap_count=len_col%graph_max
    if leap_count>0: loop_count+=1
    
    for loop in range(loop_count):
        
        if (loop_count-1 == loop) & (leap_count != 0) : graph_max=leap_count
        if loop_count==1 : g_height=4
        #print(graph_max,len_col,loop_count,leap_count)
        fig, axes = plt.subplots(graph_max, n_cols, figsize=(15, g_height*len(cols)),squeeze=True)
        fig.patch.set_facecolor('#232425')
        #plt.rcParams['figure.facecolor']='#232425'
        
        
        stt=set([True if i==target else False for i in num_col])
        
        count_l=0
        sep_first=3*loop
        sep_last=3*loop+3
    
            
        
        if len(stt)>1:
            for i, var_name in enumerate(cols[sep_first:sep_last],start=1):
                count_l+=1
                if len(cols)==count_l: sep_last=len(cols)
                i-=1
                if var_name==target:
                    t_vc=df[target].value_counts()
                    grouped=pd.DataFrame(list(zip(t_vc.values,t_vc.index.to_list())),columns=[var_name,'values'])
                    grouped.columns=[var_name,'values']
    
                else:
                    grouped=df.groupby([var_name],as_index=False)[target].count()
                    grouped.columns=[var_name,'values']
                    
                
                
                if grouped[var_name].value_counts().shape[0]<=6:
                    if graph_max==1:
                        _, wedges, autotexts = axes[0].pie(grouped['values'],autopct='%1.1f%%', startangle=10, colors=sns.color_palette('dark'),
                                                    textprops={'fontsize': 10,'va':"center",'rotation_mode' : 'anchor'},rotatelabels=True,wedgeprops={"linewidth":1.5,"edgecolor":"white"})
                        axes[0].legend(labels=grouped[var_name], loc='upper center',bbox_to_anchor=(0.5, 0.08), ncol=3,
                                         facecolor='#232425',edgecolor='#232425')
                        axes[0].set_title(var_name,pad=10)
                    else:
                        _, wedges, autotexts = axes[i,0].pie(grouped['values'],autopct='%1.1f%%', startangle=10, colors=sns.color_palette('dark'),
                                                    textprops={'fontsize': 10,'va':"center",'rotation_mode' : 'anchor'},rotatelabels=True,wedgeprops={"linewidth":1.5,"edgecolor":"white"})
                        axes[i,0].legend(labels=grouped[var_name], loc='upper center',bbox_to_anchor=(0.5, 0.08), ncol=3,
                                         facecolor='#232425',edgecolor='#232425')
                        axes[i,0].set_title(var_name,pad=10)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(9.5)
                else:
                    if graph_max==1:
                        ax=sns.stripplot(x=var_name, y=target, data=df,ax=axes[0],palette='bright')
                        axes[0].set_title(var_name,pad=10)
                        axes[0].set_facecolor('#232425')
                    else:
                        ax=sns.stripplot(x=var_name, y=target, data=df,ax=axes[i,0],palette='bright')
                        axes[i,0].set_title(var_name,pad=10)
                        axes[i,0].set_facecolor('#232425')
                    ax.set_xlabel('')
                    ax.spines['left'].set_lw(1)
                    ax.spines['top'].set_lw(0)
                    ax.spines['right'].set_lw(0)
                    ax.spines['bottom'].set_lw(1)
                    ax.grid(color='w',linestyle='-.',linewidth=0.3)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right',size=9)
                
                
                
                
                #sns.set_style("whitegrid", {'grid.linestyle': '--'})
                if graph_max==1:
                    ax=sns.countplot(x=var_name,data=df, ax=axes[1],palette=sns.color_palette('dark'))
                    axes[1].set_facecolor('#232425')
                    axes[1].set_title(var_name,pad=10)
                else:
                    ax=sns.countplot(x=var_name,data=df, ax=axes[i, 1],palette=sns.color_palette('dark'))
                    axes[i, 1].set_facecolor('#232425')
                    axes[i,1].set_title(var_name,pad=10)
                
                ax.bar_label(ax.containers[0], fmt='%.1f',color='#bfff00',fontsize='9')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right',size=9)
                ax.set_yticklabels(ax.get_yticklabels(),size=9)
                ax.set_xlabel('')
                ax.spines['left'].set_color('white')        
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_lw(1)
                ax.spines['top'].set_lw(0)
                ax.spines['right'].set_lw(0)
                ax.spines['bottom'].set_lw(1)
                ax.grid(color='w',linestyle='-.',linewidth=0.3)
                
            plt.subplots_adjust(bottom=0.8, hspace=0.9)
            plt.tight_layout()
            plt.show()
        else:
            for i, var_name in enumerate(cols,start=1):
                i-=1
                if var_name==target:
                    t_vc=df[target].value_counts()
                    grouped=pd.DataFrame(list(zip(t_vc.values,t_vc.index.to_list())),columns=['values',var_name])
                    grouped.columns=['values',var_name]
    
                else:
                    grouped=df.groupby([var_name],as_index=False)[target].count()
                    grouped.columns=[var_name,'values']
                    
                
                
                
                
                if grouped[var_name].value_counts().shape[0]<=6:
                    if graph_max==1:
                        _, wedges, autotexts = axes[0].pie(grouped['values'],autopct='%1.1f%%', startangle=10, colors=sns.color_palette('dark'),
                                                    textprops={'fontsize': 10,'va':"center",'rotation_mode' : 'anchor'},rotatelabels=True,wedgeprops={"linewidth":1.5,"edgecolor":"white"})
                        axes[0].legend(labels=grouped[var_name], loc='upper center',bbox_to_anchor=(0.5, 0.08), ncol=3,
                                         facecolor='#232425',edgecolor='#232425')
                        axes[0].set_title(var_name,pad=10)
                    else:
                        _, wedges, autotexts = axes[i,0].pie(grouped['values'],autopct='%1.1f%%', startangle=10, colors=sns.color_palette('dark'),
                                                    textprops={'fontsize': 10,'va':"center",'rotation_mode' : 'anchor'},rotatelabels=True,wedgeprops={"linewidth":1.5,"edgecolor":"white"})
                        axes[i,0].legend(labels=grouped[var_name], loc='upper center',bbox_to_anchor=(0.5, 0.08), ncol=3,
                                         facecolor='#232425',edgecolor='#232425')
                        axes[i,0].set_title(var_name,pad=10)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(9.5)
                    
                else:
                    if graph_max==1:
                        ax=sns.stripplot(x=var_name, y=target, data=df,ax=axes[0],palette='bright')
                        axes[0].set_title(var_name,pad=10)
                        axes[0].set_facecolor('#232425')
                    else:
                        ax=sns.stripplot(x=var_name, y=target, data=df,ax=axes[i,0],palette='bright')
                        axes[i,0].set_title(var_name,pad=10)
                        axes[i,0].set_facecolor('#232425') 
                    ax.set_xlabel('')
                    ax.spines['left'].set_lw(1)
                    ax.spines['top'].set_lw(0)
                    ax.spines['right'].set_lw(0)
                    ax.spines['bottom'].set_lw(1)
                    ax.grid(color='w',linestyle='-.',linewidth=0.3)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right',size=9)
                    
              
                
                
                
                if graph_max==1:
                    ax=sns.barplot(x=var_name, y='values', data=grouped, ax=axes[1],palette=sns.color_palette('dark'))
                    ax.bar_label(ax.containers[0], fmt='%.1f',color='#bfff00',fontsize='9')
                    axes[1].set_facecolor('#232425')
                    axes[1].set_title(var_name,pad=10)
                else:
                    ax=sns.barplot(x=var_name, y='values', data=grouped, ax=axes[i, 1],palette=sns.color_palette('dark'))
                    ax.bar_label(ax.containers[0], fmt='%.1f',color='#bfff00',fontsize='9')
                    axes[i, 1].set_facecolor('#232425')
                    axes[i,1].set_title(var_name,pad=10)
                
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right',size=9)
                ax.set_yticklabels(ax.get_yticklabels(),size=9)
                ax.set_xlabel('')
                
                ax.spines['left'].set_color('white')        
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_lw(1)
                ax.spines['top'].set_lw(0)
                ax.spines['right'].set_lw(0)
                ax.spines['bottom'].set_lw(1)
                
                ax.grid(color='w',linestyle='-.',linewidth=0.3)
                
            fig.subplots_adjust(bottom=0.2, hspace=0.5)
            plt.tight_layout()
            plt.show()
        count_l=0
   
    
def pairplot(df,title,n_cols=3,drop=[],render_local=False):
    """
    Draws the distrubiton graph of continuous values ​​with box plot.


    Parameters
    ----------
    df : DataFrame
    title : String
        Figure Title.
    n_cols : Int, optional
        Subplots column number. The default is 3.
    drop : List, optional
       If you want drop any feature , you have to defined list. The default is [].
    render_local: Bool
        Plotly can't run the local IDE. If change True,render open the default browser.

    """
    if render_local==True: io.renderers.default = 'browser'
    else: io.renderers.default = 'notebook_connected'
    date_cols,num_but_cat,cat_col,cat_but_num,num_col=dedect_features(df,20,30)
    colors_d = '#E48F72'
    cols=df[num_col].columns.drop(drop).to_list()
    n_rows=((len(cols) - 1) // n_cols + 1)*2
    
    cnt=0
    title_c=0
    sub_title=[]
    for i in range(0,len(cols)*2-1):
        if (cnt>2) & (cnt<6):
            sub_title.append("")
            cnt+=1
            if cnt==6: 
                cnt=0
            continue
        else:
            if len(cols)-1>=title_c:
                sub_title.append("<span style='color:#d61c38;font-weight:bold;'>{}</span>".format(cols[title_c]))
                title_c+=1
                cnt+=1
            else:
                break
    fig= make_subplots(rows=n_rows,cols=n_cols,subplot_titles=(sub_title))
    col_n=0
    row_n=1
    colors_list = px.colors.qualitative.Alphabet
    selected_elements=[]
    for i, var_name in enumerate(cols,start=1):
        def random_element_without_duplicates(list):
            """
            Return a random element from the list without duplicates.# noqa:E501
          
            Args:
              list: The list of elements.
          
            Returns:
              A random element from the list.
            """
            
            random_index = random.randint(0, len(list) - 1)
            while list[random_index] in selected_elements:
                random_index = random.randint(0, len(list) - 1)
            selected_elements.append(list[random_index])
            
            return list[random_index]
        
        random_color = random_element_without_duplicates(colors_list)
        
        sleep(0.25)
        col_n=col_n+1
        if col_n==n_cols+1:
            col_n=1
            row_n=row_n+2
        if row_n==n_rows+1:
            row_n=1
            
        boxfig= go.Figure(data=[go.Box(x=df[var_name], showlegend=False, notched=True, marker_color=random_color, name=var_name)])
        for k in range(len(boxfig.data)):
            fig.add_trace(boxfig.data[k], row=row_n, col=col_n)
        group_labels = [var_name]
        hist_data=[df[var_name].values]
        distplfig = ff.create_distplot(hist_data, group_labels, colors=[random_color],
                     bin_size=.2, show_rug=False)
        for k in range(len(distplfig.data)):
            fig.add_trace(distplfig.data[k],
            row=row_n+1, col=col_n)
    fig.update_layout(barmode='overlay',
                     title={'text':f'<b>{title} Distrubitions</b>',
                             'font_size':24,
                             'font_color': 'Red',
                             'y':0.98,
                             'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'}
                     ,showlegend=False,height=1400,width=1200)
    fig.update_yaxes(showticklabels=False)
    colors_list = px.colors.qualitative.Alphabet
    fig.show()
  
    
def outlier_dedection(df,target,model,drop='',contaminations=[0.0001,0.0005,0.001,0.005,0.01,0.07,0.1,0.15,0.25],render_local=False,graph=False):
    """
    Function using Isolation Forest method and find the outlier data.# noqa:E501

    Parameters
    ----------
    df : DataFrame
        
    target : String
        Target Feature in DataFrame
    drop : List
        If you want drop any feature , you have to defined list.
    model : Object Funciton
        Define estimator for isolation forest.
    contaminations : List, optional
        The value that must be defined for outlier precision in the isolation forest. The default is [0.0001,0.0005,0.001,0.005,0.01,0.07,0.1,0.15,0.25].# noqa:E501
    graph : Bool, optional
        If you want draw contamination value of features,change statment to 'True' "". The default is False.# noqa:E501
    render_local: Bool
        Plotly can't run the local IDE. If change True,render open the default browser.
    Returns
    -------
    list

    """
    
    
    
    n_cols=3
    from sklearn.ensemble import IsolationForest
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.decomposition import PCA
    cols=df.columns
    if drop!='':
        cols=df.columns.drop(drop)
    anomaly_dict={}
    n_rows=((len(cols) - 1) // n_cols + 1)
    col_n=0
    row_n=1
    fig= make_subplots(rows=n_rows,cols=n_cols,subplot_titles=(cols))
    def evaluate_outlier_classifier(model,data):
        labels=model.fit_predict(data)
        return data[labels==1],data[labels==-1]
    def evaluate_regressor(model,inliers,target):
        X=inliers.drop(target,axis=1)
        y=inliers[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        return round(rmse, 3)
    
    scores = dict()
    for c in contaminations:
        # Instantiate IForest with the current c
        iforest = IsolationForest(n_estimators=100,max_samples='auto',max_features=1.0,bootstrap=False,contamination=c, random_state=42,verbose=0)

        # Get inliers with the current IForest
        inliers,outliers = evaluate_outlier_classifier(iforest, df[cols])

        # Calculate and store RMSE into scores
        rmse = evaluate_regressor(model,inliers,target)
        scores[c]=rmse

    print(scores)
    best_value = max(scores, key=scores.get)
    inliers,outliers = evaluate_outlier_classifier(iforest, df[cols])
    
    
    
    if graph==True:
        if render_local==True: io.renderers.default = 'browser'
        else: io.renderers.default = 'notebook_connected'
        pca = PCA(2)
        pca.fit(df[cols])
        res=pd.DataFrame(pca.transform(df[cols]))
        plt.title("IsolationForest with PCA 2-Dimensions")
        
        b1 = plt.scatter(res[0], res[1], c='green',
                         s=20,label="normal points")
        b1 =plt.scatter(res.iloc[outliers.index.to_list(),0],res.iloc[outliers.index.to_list(),1], c='green',s=20,  edgecolor="red",label="predicted outliers")
        plt.legend(loc="upper right")
        plt.autoscale() 
        
        
        for col in cols:
            def setColor(val):
                if ([True for i in outliers.index.to_list() if i==val]):

                    return "red"
                else:
                    return "green"
            col_n=col_n+1
            if col_n==n_cols+1:
                col_n=1
                row_n=row_n+1
            if row_n==n_rows+1:
                row_n=1
            fig.add_trace(go.Scatter(
                            y=df[col],
                            mode='markers',
                            opacity=0.7,
                            marker=dict(color=[setColor(val) for val in df[col].index])
                        ),
                        row=row_n,col=col_n    
                              )
        fig.update_layout(barmode='overlay',
                        title={'text':'<b>Train_df Outliers with Isolation Forest</b>',
                                 'font_size':20,
                                 'font_color': 'Red',
                                 'y':0.98,
                                 'x':0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'},showlegend=False,height=1200,width=1000)
        fig.show()
        
        plt.show()
        
        
        
    return outliers

