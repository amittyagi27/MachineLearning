import oml
import pandas as pd

#Connect to Database and pull data
oml.connect(user='oml_user1', password='oracle', host='localhost', port=1521, service_name='orclpdb19', automl=True)
print(oml.isconnected())

# Create training and test data.
dataset=oml.sync(table='CUSTOMER_CHURN_HISTORY')
dataset
train, test = dataset.split(ratio=(0.8, 0.2), hash_cols='CUSTOMERID',
                       seed=32)
x_train, y_train = train.drop('EXITED'), train['EXITED']
x_train

# Train & Test the ML Model
model=oml.glm("classification")


##Train the model
model.fit(x_train,y_train, case_id='CUSTOMERID')

# Return the prediction probability.
model.predict(test.drop('EXITED'), 
               supplemental_cols = test[:, ['SURNAME', 
                                                'CREDITSCORE', 
                                                'GEOGRAPHY',
                                               'GENDER',
                                               'AGE',
                                               'TENURE',
                                               'BALANCE',
                                               'NUMOFPRODUCTS',
                                               'HASCRCARD',
                                               'ISACTIVEMEMBER',
                                               'CUSTOMERID']],proba = True)

model.score(test.drop('EXITED'), test[:, ['EXITED']])

# Do prediction on current data and Write back prediction in database using OML cursor
current_dataset=oml.sync(table='CUSTOMER_CHURN_CURRENT')

current_prediction=model.predict(current_dataset.drop('EXITED'), 
               supplemental_cols = current_dataset[:, ['SURNAME', 
                                                'CREDITSCORE', 
                                                'GEOGRAPHY',
                                               'GENDER',
                                               'AGE',
                                               'TENURE',
                                               'BALANCE',
                                               'NUMOFPRODUCTS',
                                               'HASCRCARD',
                                               'ISACTIVEMEMBER',
                                               'CUSTOMERID']],proba = True)

oml_current_prediction=current_prediction.pull()

oml_current_prediction_obj = oml.create(oml_current_prediction, table = 'CUSTOMER_CHURN_CURRENT_OML')
