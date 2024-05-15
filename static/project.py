# # IMPORTING THE DEPENDENCIES
#-----------------------------
# pip install numpy pandas scikit-learn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# #DATA COLLECTION AND PROCESSING
# #-------------------------------
heart_data=pd.read_csv('heart_disease_data.csv')

# print(heart_data.head())
# print(heart_data.tail())
# print(heart_data.shape)


# #GETTING INFO ABOUT DATA
# #------------------------
# print(heart_data.info())


# #CHECKING FOR MISSING VALUES
# #---------------------------
# print(heart_data.isnull().sum())


# #STATISTICAL MEASURE ABOUT DATA
# #------------------------------
# print(heart_data.describe())


# #CHECKING DISTRIBUTION OF TARGET VARIABLE
# #-----------------------------------------
print(heart_data['target'].value_counts())


# #Splitting features and target
# #------------------------------
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
print(X)
print(Y)


# # #Splitting the data into training and test data
# # #----------------------------------------------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)


# # #MODEL TRAINING
# # #---------------
# # #Logistic Regression : 
model2=LogisticRegression(max_iter=100000)

model2.fit(X_train,Y_train)



# # #Model evaluation
# # #--------------------
# # #accuracy score
X_train_prediction=model2.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
# # print('Accuracy on training data : ',training_data_accuracy)

X_test_prediction=model2.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
# # print('Accuracy on test data : ',test_data_accuracy)


from sklearn.metrics import roc_curve, roc_auc_score

# Predict probabilities for the positive class (class 1) from the test set
y_probabilities = model2.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(Y_test, y_probabilities)

# Calculate the area under the ROC curve (AUC)
auc = roc_auc_score(Y_test, y_probabilities)

# Plot the ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()




# # #Building a predictive system
# # #-------------------------
# input_data=(57,0,0,120,354,0,1,163,1,0.6,2,0,2)

# # input_data_numpy=np.asarray(input_data)

# # input_data_reshape=input_data_numpy.reshape(1,-1)
# # prediction=model2.predict(input_data_reshape)
# # print(prediction)

# # if(prediction[0]==0):
# #     print("person doesnot have heart disease")
# # else:
# #     print("person have heart disease")

import pickle
pickle.dump(model2,open("project.pkl","wb"))
