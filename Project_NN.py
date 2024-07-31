import pandas as pd
import scipy as scp
import numpy as np
import time
from copy import deepcopy
import seaborn as sns
from matplotlib import pyplot as plt
import imblearn 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFpr,SelectKBest
import warnings
warnings.filterwarnings('ignore')
#-----------------------------++++++++++++++++++++++++++++++++++++++++++++++++++++------------------------------------------------------------------------------
df_main = pd.read_csv('Ultrasound_1.csv')
df = deepcopy(df_main)
print(df['Pass/Fail'].value_counts()/df.shape[0])
y = df.pop('Pass/Fail').values
print(type(y))

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
sel_feat =SelectFpr(alpha = 0.05)
df_new = sel_feat.fit_transform(df_new,y)
x = df_new
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
X_os = imblearn.over_sampling.ADASYN(sampling_strategy= 0.30 , n_neighbors=8)
X_train,y_train = X_os.fit_resample(X_train,y_train)
std_sc = StandardScaler()
X_train= std_sc.fit_transform(X_train)
X_test= std_sc.fit_transform(X_test)

#-------------------------------<><><><><><><><><><><><><><><><><>>--------------------------------------------------------------------------
###################---------------------------------------------------------------------##################################################################
# Define the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  
    # tf.keras.layers.Dense(2048, activation='relu'), 
    # tf.keras.layers.Dropout(0.15),
    # tf.keras.layers.Dense(1024, activation='relu'),  
    # tf.keras.layers.Dropout(0.15),
    # tf.keras.layers.Dense(512, activation='relu'), 
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Dense(256, activation='relu'), 
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Dense(128, activation='relu'), 
    # #tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Dense(64, activation='relu'),
    # #tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(32, activation='relu'), 
    #tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(16, activation='relu'),  
    tf.keras.layers.Dense(1, activation='softmax')  # Output layer for binary classification
])

# Print model summary
print(model.summary())

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Fit the model
model.fit(X_train,y_train, epochs=500,validation_data = (X_test,y_test))