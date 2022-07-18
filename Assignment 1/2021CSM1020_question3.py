# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:42:47 2022

@author: Ananya_Chaudhuri
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df=pd.read_csv("insurance.csv")
df['sex'] = df['sex'].replace(['male','female'],[1,0])
df['smoker'] = df['smoker'].replace(['yes','no'],[1,0])
df['region'] =df['region'].replace(['northeast','northwest','southeast','southwest'],[1,2,3,4])
print("*********** Original Dataset ************")
print(df)
df2=df.drop(['charges'],axis=1)
#print(df2)
#print(df)
for column in df2:
    df2[column] = (df2[column] - df2[column].mean()) / df2[column].std()
mean,std=0,0
for column in df2:
    print(column," mean: ",round(df2[column].mean(),1),"std: ",round(df2[column].std(),1))
df2['charges'] = df['charges'].values
#print(df2)
df1=df2
def train_test_split(df1):
    shuffle_df1 = df1.sample(frac=1)
    train_size = int(0.8 * len(df1))
    df_train = shuffle_df1[:train_size]
    df_test = shuffle_df1[train_size:]
    return df_train,df_test
df_train,df_test=train_test_split(df1)
#print(type(df_train))

def cross_validation_split(dataset, folds):
        dataset_split = []
        df_copy = dataset
        fold_size = int(df_copy.shape[0] / folds)
        
        for i in range(folds):
            fold = []
            while len(fold) < fold_size:
                r = random.randrange(df_copy.shape[0])
                index = df_copy.index[r]
                fold.append(df_copy.loc[index].values.tolist())
                df_copy = df_copy.drop(index)
            dataset_split.append(np.asarray(fold))
            
        return dataset_split
#print(cross_validation_split(df_train,4))

def dataset_split(df):
    X = df.drop([df.columns[-1]], axis = 1)
    y = df[df.columns[-1]]
    return X, y

def mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    MSE = np.square(np.subtract(y_true, y_pred)).mean()
    return MSE

def bias_cal(y_test,y_pred):
    t=y_test.shape[0]
    y_test,y_pred=np.array(y_test).mean(),np.array(y_pred).mean()
    b=(y_pred-y_test)**2
    return b/t
def var_cal(y_test):
    y_test=np.array(y_test)
    p=y_test.mean()
    y_test=y_test-p
    var=np.square(y_test).mean()
    return var
def ridgereg(X, Y, l2):
    iterations=1000
    p,q=X.shape
    W=np.zeros(q)
    b=0
    for i in range( iterations ) :            
        a,c=update_weights(W,X,l2,Y,p,b) 
        W=a
        b=c
    return W,b
def update_weights(W,X,l2,Y,p,b):
    learning_rate=0.01
    Y_pred = predridgereg( X,W,b )
    # calculate gradients      
    dW = ( - ( 2 * ( X.T ).dot( Y - Y_pred ) ) + ( 2 * l2 * W ) ) / p     
    db = - 2 * np.sum( Y - Y_pred ) / p
    W = W -learning_rate * dW    
    b = b - learning_rate * db  
    return W,b
def predridgereg(X, weights,b):
    return X.dot( weights ) +b
folds=10
s=cross_validation_split(df_train, folds)
avg_Train_error=[]
avg_Test_error=[]
bias_list=[]
vari=[]
lambda_values=[-80,-60,-40,-20,0.01,10,20,30]
print("********* Calculation Started ********")
X_df_test,y_df_test=dataset_split(df_test)
for i in range(0,8):

    l=lambda_values[i]
    error=[]
    for split in range(len(s)):
        #print("Working on ", split+1 ," dataset: ")
        #print(s[split])
        df_split = pd.DataFrame(s[split], columns = ['age','sex','bmi','children','smoker','region','charges'])
        df_train_new=df_train[~df_train.apply(tuple,1).isin(df_split.apply(tuple,1))]
        #print(df_train_new.shape,df_split.shape)
        X_train,y_train=dataset_split(df_train_new)
        X_test,y_test=dataset_split(df_split)
        
        W,b=ridgereg(X_train,y_train,l)
        y_pred=predridgereg( X_test,W,b )
        error.append(mean_square_error(y_test, y_pred))
    #print(error)
    y_pred_test=predridgereg(X_df_test,W,b)
    vari.append(var_cal(y_df_test))
    avg_Test_error.append(mean_square_error(y_df_test,y_pred_test))
    avg_Train_error.append(sum(error)/folds)
    bias_list.append(bias_cal(y_df_test,y_pred_test))

print("********* Calculation Ended ********")
#print("Average Train error :",avg_Train_error)
#print("Average Test error :",avg_Test_error)
#print("Variance :",vari)
#print("bias**2 :",bias_list)

plt.plot(lambda_values, avg_Train_error, label ='Average Train Error')
plt.xlabel("lambda values")
plt.ylabel("Error")
plt.show()


plt.plot(lambda_values, avg_Test_error, label ='Average Test Error')
plt.xlabel("lambda values")
plt.ylabel("Mean Square Error")
plt.show()

plt.plot(lambda_values, vari, label ='Variance')
plt.xlabel("lambda values")
plt.ylabel("Variance")
plt.show()


plt.plot(lambda_values, bias_list, label ='Bias Square')
plt.xlabel("lambda values")
plt.ylabel("Bias Square")
plt.show()
