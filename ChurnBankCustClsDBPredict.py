    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:13:13 2022

@author: amittyagi
"""

# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cx_Oracle

# =============================================================================
# import data 
# =============================================================================

dsn_tns=cx_Oracle.makedsn("130.**.**.**",'**',service_name='pdb1.***')     #Update database details here.
conn=cx_Oracle.connect("dmuser",'****', dsn_tns) 
print("************Database Connected *******************") 

dataset=pd.read_sql('select * from customer_churn_history', con=conn)

# =============================================================================
# Import Data & Feature Extraction
# =============================================================================

X = dataset.iloc[:,3:13]             # Dataframe
y = dataset.iloc[:,13].values        # y array

# =============================================================================
# Encode Categorical values using get_dummies()
# =============================================================================
X = pd.get_dummies(X,columns=['GEOGRAPHY','GENDER'],drop_first=True)

X = X.values  # Dataframe to numpy array 


# =============================================================================
# Train-Test Split 80-20
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# =============================================================================
# Sclaing of Values
# =============================================================================
from sklearn.preprocessing import StandardScaler
scObj = StandardScaler()

scObj.fit(X_train)

X_train = scObj.transform(X_train)
X_test = scObj.transform(X_test)

# =============================================================================
# Binary Classification
# =============================================================================

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model=linear_model.LinearRegression()

##Train the model

model.fit(X_train,y_train)

## Test the Model

y_predict=model.predict(X_test)


#accuracy 

y_predict=np.round(abs(y_predict))  ## Change output into 0 and 1

cm=confusion_matrix(y_test, y_predict)

self_accuracy = accuracy_score(y_test,y_predict)


# Plot outputs
import matplotlib.pyplot as plt
plt.plot(y_test,c="green", lw=5)
plt.plot(y_predict, c="Red")
plt.show()


# =============================================================================
# Predict the current data
# =============================================================================

current_dataset = pd.read_sql('select * from customer_churn_current', con=conn)

x_current = current_dataset.iloc[:,3:13]             # Dataframe

x_current = pd.get_dummies(x_current,columns=['GEOGRAPHY','GENDER'],drop_first=True)
x_current = x_current.values  # Dataframe to numpy array 


scObj_current = StandardScaler()
scObj_current.fit(x_current)
x_current = scObj_current.transform(x_current)


y_pred_current = model.predict(x_current)
y_pred_current=np.round(abs(y_pred_current))

current_dataset['Churn_prediction'] = y_pred_current

# write back to file
current_dataset.to_csv('ChurnBankCustomersLRPred.csv', index=False)

cursor=conn.cursor()

sql="insert into CUSTOMER_CHURN_PRED(CHURN_PREDICTION,CUSTOMERID,AGE,BALANCE,CREDITSCORE,ESTIMATEDSALARY,GENDER,GEOGRAPHY,HASCRCARD,ISACTIVEMEMBER,NUMOFPRODUCTS,TENURE) values(:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12)"

for i in range(len(current_dataset)):
    
    cursor.execute(sql,(str(current_dataset.loc[i,"Churn_prediction"]),str(current_dataset.loc[i,"CUSTOMERID"]), str(current_dataset.loc[i,"AGE"]),
                  str(current_dataset.loc[i,"BALANCE"]),str(current_dataset.loc[i,"ESTIMATEDSALARY"]), str(current_dataset.loc[i,"GENDER"]),
                  str(current_dataset.loc[i,"GEOGRAPHY"]), str(current_dataset.loc[i,"HASCRCARD"]),str(current_dataset.loc[i,"ISACTIVEMEMBER"]),
                  str(current_dataset.loc[i,"NUMOFPRODUCTS"]),str(current_dataset.loc[i,"CREDITSCORE"]), str(current_dataset.loc[i,"TENURE"])))
    
    conn.commit()

cursor.close()
conn.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

