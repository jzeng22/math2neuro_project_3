import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from operator import add
from operator import sub

demographic = pd.read_csv(r'/Users/jennycinno/Documents/GitHub/math2neuro_project_3/demographic.csv')
ERPdata = pd.read_csv(r'/Users/jennycinno/Documents/GitHub/math2neuro_project_3/ERPdata.csv')

# creating a list of exact time stamps from 0 to 500 ms 
num_time_stamps = 3072
time_stamps = []

for i in range(0, num_time_stamps):
    if ERPdata.iloc[i, 11] >= 0 and ERPdata.iloc[i, 11] <= 500:
        time_stamps.append(ERPdata.iloc[i, 11])

# returns a list of the average ERP data for a certain group, electrode, and condition
def average_ERP(group, electrode, condition):
    if condition == 1:
        start_row = 1536
    elif condition == 2:
        start_row = 4608
        
    increment = 9216
    
    group_ERP = []
    
    for i in range(0, 81):
        if demographic.iloc[i,1] == group:
            patient_ERP = []
            for j in range(0, len(time_stamps)):
                patient_ERP.append(ERPdata.iloc[start_row + (i * increment) + j, electrode + 1])
            group_ERP.append(patient_ERP)
    
    average_ERP = [0] * len(time_stamps)
    for i in range(0, len(group_ERP)):
        average_ERP = list(map(add, average_ERP, group_ERP[i]))
    
    average_ERP = [x / len(group_ERP) for x in average_ERP]
    
    return average_ERP

# plotting data 
for electrode in range(1, 10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    fig.suptitle(f"Average ERP of Controls vs. Patients for Electrode {electrode}")
    
    ax1.set_title("Condition 1")
    ax1.plot(time_stamps, average_ERP(0, electrode, 1), label = "Controls", color = "red")
    ax1.plot(time_stamps, average_ERP(1, electrode, 1), label = 'Patients', color = "orange")
    ax1.set_xlim([0, 500])
    ax1.set_ylim([-7, 7])
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("µV")
    ax1.legend()
    
    ax2.set_title("Condition 2")
    ax2.plot(time_stamps, average_ERP(0, electrode, 2), label = "Controls", color = "red")
    ax2.plot(time_stamps, average_ERP(1, electrode, 2), label = 'Patients', color = "orange")
    ax2.set_xlim([0, 500])
    ax2.set_ylim([-7, 7])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("µV")
    ax2.legend()  
    
# feature extraction
# N100 and P200 peak amplitude for controls in condition 1
N100_controls_1 = []
P200_controls_1 = []
for electrode in range(1, 10):
    ERP = average_ERP(0, electrode, 1)
    N100_controls_1.append(abs(min(ERP)))
    P200_controls_1.append(abs(max(ERP)))
    
# N100 and P200 peak amplitude for controls in condition 2
N100_controls_2 = []
P200_controls_2 = []
for electrode in range(1, 10):
    ERP = average_ERP(0, electrode, 2)
    N100_controls_2.append(abs(min(ERP)))
    P200_controls_2.append(abs(max(ERP)))

# N100 and P200 peak amplitude for patients in condition 1
N100_patients_1 = []
P200_patients_1 = []
for electrode in range(1, 10):
    ERP = average_ERP(1, electrode, 1)
    N100_patients_1.append(abs(min(ERP)))
    P200_patients_1.append(abs(max(ERP)))
    
# N100 and P200 peak amplitude for patients in condition 2
N100_patients_2 = []
P200_patients_2 = []
for electrode in range(1, 10):
    ERP = average_ERP(1, electrode, 2)
    N100_patients_2.append(abs(min(ERP)))
    P200_patients_2.append(abs(max(ERP)))
    
# peak amplitude comparison between controls and patients 
N100_CvsP_1 = list(map(sub, N100_controls_1, N100_patients_1))
N100_CvsP_2 = list(map(sub, N100_controls_2, N100_patients_2))
P200_CvsP_1 = list(map(sub, P200_controls_1, P200_patients_1))
P200_CvsP_2 = list(map(sub, P200_controls_2, P200_patients_2))


# changes in N100 and P200 between conditions 1 and 2 (negative #s indicate suppresion in condition 1, positive #s indicate increase in condition 1)
N100_controls_1vs2 = list(map(sub, N100_controls_1, N100_controls_2))
P200_controls_1vs2 = list(map(sub, P200_controls_1, P200_controls_2))
N100_patients_1vs2 = list(map(sub, N100_patients_1, N100_patients_2))
P200_patients_1vs2 = list(map(sub, P200_patients_1, P200_patients_2))

# differences in changes in N100 and P200 between controls and patients 
N100_CvsP = list(map(sub, N100_patients_1vs2, N100_controls_1vs2)) # positive #s mean controls suppressed more in condition 1
P200_CvsP = list(map(sub, P200_controls_1vs2, P200_patients_1vs2)) # positive #s mean controls increased more in condition 1

"""
# creating a dataframe of the preliminary data analysis
row_labels = ['Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']
column_labels = ['N100 C1', 'P200 C1', 'N100 C2', 'P200 C2', 'N100 P1', 'P200 P1', 'N100 P2', 'P200 P2', 'N100 C1vs2', 'P200 C1vs2', 'N100 P1vs2', 'P200 P1vs2', 'N100 Supression CvsP', 'P100 Increase CvsP']
data = np.column_stack((N100_controls_1, P200_controls_1, N100_controls_2, P200_controls_2, N100_patients_1, P200_patients_1, N100_patients_2, P200_patients_2, N100_controls_1vs2, P200_controls_1vs2, N100_patients_1vs2, P200_patients_1vs2, N100_CvsP, P200_CvsP))
data_analysis_df = pd.DataFrame(data, index = row_labels, columns = column_labels)
data_analysis_df.to_csv('/Users/jennycinno/Documents/GitHub/math2neuro_project_3/data_analysis.csv')
"""

# creating a dataframe of the preliminary data analysis
row_labels = ['Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']
column_labels = ['N100 C1 vs P1', 'N100 C2 vs P2', 'P200 C1 vs P1', 'P200 C2 vs P2','N100 Supression C vs P', 'P100 Increase C vs P']
data = np.column_stack((N100_CvsP_1, N100_CvsP_2, P200_CvsP_1, P200_CvsP_2, N100_CvsP, P200_CvsP))
data_analysis_df = pd.DataFrame(data, index = row_labels, columns = column_labels)
data_analysis_df.to_csv('/Users/jennycinno/Documents/GitHub/math2neuro_project_3/data_analysis.csv')

