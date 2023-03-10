import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
# from xgboost import cv

demographic = pd.read_csv(r'/Users/jennycinno/Documents/GitHub/math2neuro_project_3/demographic.csv')
ERPdata = pd.read_csv(r'/Users/jennycinno/Documents/GitHub/math2neuro_project_3/ERPdata.csv')

# creating a list of exact time stamps from 0 to 500 ms 
num_time_stamps = 3072
time_stamps = []

for i in range(0, num_time_stamps):
    if ERPdata.iloc[i, 11] >= 0 and ERPdata.iloc[i, 11] <= 500:
        time_stamps.append(ERPdata.iloc[i, 11])

# returns a list representing the ERP data for a certain patient, electrode, and condition
def patient_ERP(patient, electrode, condition):
    if condition == 1:
        start_row = 1536
    elif condition == 2:
        start_row = 4608
        
    increment = 9216
    
    patient_ERP = []
    for i in range(0, len(time_stamps)):
        patient_ERP.append(ERPdata.iloc[start_row + (patient * increment) + i, electrode + 1])
            
    return patient_ERP

"""
N100_1_C = []
N100_1_P = []
for i in range(0, 81):
    if demographic.iloc[i, 1] == 0:
        N100_1_C.append(min(patient_ERP(i, 1, 1)))
    elif demographic.iloc[i, 1] == 1:
        N100_1_P.append(min(patient_ERP(i, 1, 1)))

plt.boxplot([N100_1_C, N100_1_P])
"""
"""
# N100 peak amplitude in condition 1, electrode 1
N100_1 = []
for i in range(0, 81):
    N100_1.append(min(patient_ERP(i, 1, 1)))
    
# N100 peak amplitude in condition 2, electrode 1
N100_2 = []
for i in range(0, 81):
    N100_2.append(min(patient_ERP(i, 1, 2)))

# P200 peak amplitude in condition 1, electrode 1
P200_1 = []
for i in range(0, 81):
    P200_1.append(max(patient_ERP(i, 1, 1)))
    
# P200 peak amplitude in condition 2, electrode 1
P200_2 = []
for i in range(0, 81):
    P200_2.append(max(patient_ERP(i, 1, 2)))
    
# N100 suppression in electrode 1
N100_suppression = []
for i in range(0, 81):
    N100_suppression.append(abs(min(patient_ERP(i, 1, 2))) - abs(min(patient_ERP(i, 1, 1))))
"""

# boxplot of N100 suppression in electrode 1 for controls vs. patients 
N100_suppression_C = []
N100_suppression_P = []
for patient in range(0, 81):
    if demographic.iloc[patient, 1] == 0:
        N100_suppression_C.append(abs(min(patient_ERP(patient, 1, 2))) - abs(min(patient_ERP(patient, 1, 1))))
    elif demographic.iloc[patient, 1] == 1:
        N100_suppression_P.append(abs(min(patient_ERP(patient, 1, 2))) - abs(min(patient_ERP(patient, 1, 1))))

plt.boxplot([N100_suppression_C, N100_suppression_P])
plt.xticks(ticks = [1, 2], labels = ['Controls', 'Patients'])
plt.title('N100 Suppression in Electrode 1')

# feature extraction
x = []
for patient in range(0, 81):
    patient_x = []
    for electrode in range(1, 10):
        patient_x.append(min(patient_ERP(patient, electrode, 1))) # N100 peak amplitude in condtion 1
        patient_x.append(min(patient_ERP(patient, electrode, 2))) # N100 peak amplitude in condtion 2
        patient_x.append(max(patient_ERP(patient, electrode, 1))) # P200 peak amplitude in condtion 1
        patient_x.append(max(patient_ERP(patient, electrode, 2))) # P200 peak amplitude in condtion 2
        patient_x.append(abs(min(patient_ERP(patient, electrode, 2))) - abs(min(patient_ERP(patient, electrode, 1)))) # N100 suppresion in condition 1 compared to condition 2
    x.append(patient_x)

y = []
for patient in range(0, 81):
    y.append(demographic.iloc[patient, 1])
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# logistic regression model
lr_model = LogisticRegression(max_iter = 200, penalty = 'l2')
cv = KFold(n_splits = 10, random_state = 1, shuffle = True)
lr_scores = cross_val_score(lr_model, x, y, scoring = 'accuracy', cv = cv)
print('Logistic Regression Model Accuracy on Test Data: %.3f' % np.mean(lr_scores))

lr_model.fit(x_train, y_train)
print('Logistic Regression Model Accuracy on Training Data: %.3f' % lr_model.score(x_train, y_train))

# xgboost model 
params = {'colsample_bytree':0.3, 'max_depth':3, 'subsample':0.5, 'gamma':5, 'eta':0.1}
xgb_model = XGBClassifier(**params)
xgb_scores = cross_val_score(xgb_model, x, y, scoring = 'accuracy', cv = cv)
print('XGBoost Model Accuracy on Test Data: %.3f' % np.mean(xgb_scores))

xgb_model.fit(x_train, y_train)
print('XGBoost Model Accuracy on Training Data: %.3f' % xgb_model.score(x_train, y_train))



