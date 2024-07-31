import pandas as pd
import scipy as scp
import numpy as np
import time
from copy import deepcopy
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report,make_scorer,confusion_matrix,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest,SelectFpr
from sklearn.pipeline import Pipeline
import imblearn 
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
#-----------------------------++++++++++++++++++++++++++++++++++++++++++++++++++++------------------------------------------------------------------------------
df_main = pd.read_csv('Ultrasound_1.csv')
df_main.loc[df_main['Pass/Fail'] == -1] = 0
df = deepcopy(df_main)
print(df['Pass/Fail'].value_counts()/df.shape[0])
y = df.pop('Pass/Fail').values


class Processing_Df:
    def __init__(self,df_lcl):
        self.df_cl = df_lcl
        self.lst_fl = []
        self.lst_int = []
        self.lst_o = []
        self.stats_lst = []
        self.lst_d_m_c = []
        
#@@@@@@@@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def df_num(self):
        #self.df_cl = self.Drop_mis_column()
        for cols in self.df_cl:
            if self.df_cl[cols].dtypes == 'float':
                self.lst_fl.append(cols)
            if self.df_cl[cols].dtypes == 'int':
                self.lst_int.append(cols)
            if self.df_cl[cols].dtypes == 'object':
                self.lst_o.append(cols)    
            
        return (self.lst_fl,self.lst_int,self.lst_o)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def df_std_removal(self):
        
        df_stats = self.df_cl.describe()
        for i in df_stats.columns:
            if df_stats.loc['std',i] == 0:
                self.stats_lst.append(i)
        self.df_cl.drop(columns=self.stats_lst,inplace=True) 
        print(self.df_cl.shape,2)
        return (self.df_cl)
#-------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def Drop_mis_column(self):
        dct_df = {cols:(self.df_cl[cols].isnull().sum()/self.df_cl.shape[0])*100 for cols in df.columns}
        for j in dct_df.keys():
            if dct_df[j] > 25.0:
                self.lst_d_m_c.append(j)
        self.df_cl.drop(columns=self.lst_d_m_c,inplace=True)
        print(self.df_cl.shape,1)
        return (self.df_cl)
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    def Standardisation(self):
        tmp = []
        for i in  self.df_cl.columns:
            if self.df_cl[i].max()>3:
                tmp.append(i)
            else:  
                continue
        std_sc = StandardScaler()
        self.df_cl[tmp] = std_sc.fit_transform( self.df_cl[tmp])
        return  self.df_cl
# #-----------------------><><><><><><><><<><>-------------------------------------------------------------
    def Mis_Treat(self):
        for i in self.df_cl.columns:
            if self.df_cl[i].isnull().sum() != 0:
                if (self.df_cl[i].skew() > 1) | (self.df_cl[i].skew() < -1):
                    self.df_cl[i].fillna(value=self.df_cl[i].median(),inplace=True)
                elif (self.df_cl[i].skew() < 1) | (self.df_cl[i].skew() > -1):
                     self.df_cl[i].fillna(value=self.df_cl[i].mean(),inplace=True)
        print(self.df_cl.shape,3)
        return self.df_cl
#--------------------------->>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<------------------------------------------
    def Replacing_outliers(self,tmp):
        df_tmp_zscore = scp.stats.zscore(self.df_cl)
        
        for i in df_tmp_zscore.columns:
            if tmp > 0:
                if (df_tmp_zscore[i]>tmp).any():
                    arr1 = np.where(df_tmp_zscore[i]>tmp)
                    
                    tmp1 = tmp*self.df_cl[i].std() + self.df_cl[i].mean()
                    for j in arr1[0]:
                        self.df_cl.loc[j,i] =  tmp1
            else:
                if (df_tmp_zscore[i]<tmp).any():
                    arr1 = np.where(df_tmp_zscore[i]<tmp)
                   
                    tmp1 = -tmp*self.df_cl[i].std() + self.df_cl[i].mean()
                    for j in arr1[0]:
                        
                        self.df_cl.loc[j,i] =  tmp1
        print(self.df_cl.shape,4)
        return self.df_cl
#-------------------------------------<><><><><><><><><><><><><><><><><>-----------------------------------------------------------------------------
    def Drop_corr_cols(self):
        ad = self.df_cl.corr().abs()
        upper = ad.where(np.triu(np.ones(ad.shape), k=1).astype(bool))
        threshold = 0.9  # Set your threshold value
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.df_cl.drop(columns=to_drop,inplace  = True)
        print(self.df_cl.shape,5)
        return self.df_cl

#-------------------------><><><><MAIN><MAIN><><><><><><><>><><><><><><><><---------------------------------------------            
    def __call__(self):
        self.Drop_mis_column()
        self.df_std_removal()
        self.Mis_Treat()
        self.Replacing_outliers(3.0)
        self.Replacing_outliers(-3.0)
        #self.Standardisation()
        self.Drop_corr_cols()
        return self.df_cl 
#-------------------------------------------------------------+++++++++++++++++-----------------------------------------------------------------------

df_o = Processing_Df(df)

#l_f,l_int,l_o = df_o.df_num()
#df_new  =  df_o.main()
df_new = df_o()
#-------------------------<><><><><><><><><><><><><><><><><<><><><><><>------------------------------------------------------------------------------
#----------------------------------<><><><><><><><><><><><><><><><><>-------------------------------------------------------------------------------
x = df_new
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.8,random_state=42)
#-------------------------------<><><><><><><><><><><><><><><><><>>--------------------------------------------------------------------------
#X_OS = imblearn.over_sampling.SMOTE(k_neighbors=8,sampling_strategy = 0.45)
X_OS_A = imblearn.over_sampling.ADASYN(sampling_strategy=0.35,n_neighbors=10)
#X_OS_A = imblearn.over_sampling.ADASYN()
#X_train, y_train = X_OS_A.fit_resample(X_train,y_train)
# y_res = deepcopy(y_train)
# y_res = pd.DataFrame(pd.Series(y_res),index = np.arange(0,y_res.shape[0]),columns = ['target'])
# #----------------------------------------------------------------------------------------------------------------------------------------------------
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
Sp_Vc_Mc = svm.SVC()
sel_feat =SelectFpr() 
#------------------------------------<><><><><><><><><><><><><><><><><><><><><><><-----------------------------------------------------------------------------------------------------
learning_rate_range = np.arange(0.01, 1, 0.05)
xgb_classifier = xgb.XGBClassifier(eta = 2.5)
model_boosting = xgb_classifier.fit(X_train, y_train)
y_tr_pred = model_boosting.predict(X_train)
y_tst_pred = model_boosting.predict(X_test)
print(precision_score(y_train,y_tr_pred))
print(precision_score(y_test,y_tst_pred))
print(recall_score(y_train,y_tr_pred))
print(recall_score(y_test,y_tst_pred))
print(confusion_matrix(y_train,y_tr_pred))
print(roc_auc_score(y_test,y_tst_pred))
# #------------------------------<><><><><><><><><><><><><><><><><><>-----------------------------------------------------------------------------------

# # with open('Evaluation.txt','w') as f1:
# #     f1.write(f'Best_Estimator -> {model_best}\n')
# #     f1.write(f'Best_Params -> {model_hy_tun.best_params_}\n')
# #     f1.write(f'Best_Score -> {model_hy_tun.best_score_}\n')
# #     f1.write(f'Best_Parameters -> {model_hy_tun.best_params_}\n')
# #     f1.write(f'the training set accuracy -> {accuracy_score(y_train,y_tr_pred)}\n')
# #     f1.write(f'the test set accuracy -> {accuracy_score(y_test,y_tst_pred)}\n')
# #     f1.write(f'the training set precsion -> {precision_score(y_train,y_tr_pred)}\n')
# #     f1.write(f'the test set precsion -> {precision_score(y_test,y_tst_pred)}\n')
# #     f1.write(f'the training set recall -> {recall_score(y_train,y_tr_pred)}\n')
# #     f1.write(f'the test set recall -> {recall_score(y_test,y_tst_pred)}\n')
# #     f1.write(f'the training F1 -> {f1_score(y_train,y_tr_pred)}\n')
# #     f1.write(f'the test F1 -> {f1_score(y_test,y_tst_pred)}\n:')
# #     f1.write(f'the training ROC -> {roc_auc_score(y_train,y_tr_pred)}\n')
# #     f1.write(f'the test ROC -> {roc_auc_score(y_test,y_tst_pred)}\n:')
# #     f1.write(f'the test -> {pd.Series(y_test).value_counts()}\n:')
# #     f1.write(f'the train -> {pd.Series(y_train).value_counts()}\n:')


# # #------------------------------------*****************************************-------------------------------------------------
