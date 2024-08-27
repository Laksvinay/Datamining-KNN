#!/usr/bin/env python
# coding: utf-8

# In[2]:


from platform import python_version
print("\n python version for K-NN classification analysis is ",python_version())


# In[3]:


import os
import pandas as pd
import numpy as np
dmdata=pd.read_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Dataset/medical_clean.csv")
dmdata.head()


# In[4]:


dmdata.info()


# In[5]:


# Rearrange dataset according to the dictionary
# Renaming survey variables in a meaningful way
dmdata.rename(columns={
    "Item1": "Timely_admission",
    "Item2": "Timely_treatment",
    "Item3": "Timely_visits",
    "Item4": "Reliability",
    "Item5": "Treatment_hours",
    "Item6": "Options",
    "Item7": "Courteous_staff",
    "Item8": "Active_listening"
}, inplace=True)

print(dmdata.columns)


# In[6]:


# Check for null values
(dmdata.isna().any())


# In[7]:


# Check for duplicates
(dmdata.duplicated().sum())


# In[8]:


categorical_features = dmdata.select_dtypes(include=['object', 'category']).columns
for feature in categorical_features:
    unique_values = dmdata[feature].unique()
    print(f"Unique values for '{feature}':")
    print(unique_values)


# In[9]:


from sklearn.preprocessing import LabelEncoder
# Label encoding for binary categorical variables
label_encoder = LabelEncoder()
binary_vars = ['ReAdmis','Soft_drink','HighBlood','Diabetes','Hyperlipidemia','Stroke','Overweight','Arthritis','BackPain','Anxiety','Allergic_rhinitis','Reflux_esophagitis','Asthma']
for var in binary_vars:
    dmdata[var] = label_encoder.fit_transform(dmdata[var])
    


# In[10]:


dmdata[binary_vars]


# In[11]:


dmdata.info()


# In[12]:


categorical_columns = dmdata.select_dtypes(include=['object']).columns.tolist()

("Remaining Categorical columns identified:", categorical_columns)


# In[13]:


#removing whitespace
dmdata['Marital']= dmdata['Marital'].str.replace(' ','')
dmdata['Job']= dmdata['Job'].str.replace(' ','')
dmdata['Initial_admin']= dmdata['Initial_admin'].str.replace(' ','')
dmdata['Complication_risk']= dmdata['Complication_risk'].str.replace(' ','')
dmdata['Services']= dmdata['Services'].str.replace(' ','')
dmdata['City']= dmdata['City'].str.replace(' ','')
dmdata['County']= dmdata['County'].str.replace(' ','')
dmdata['State']= dmdata['State'].str.replace(' ','')


# In[14]:


# To count the occurrences of each unique value
dmdata['Marital'].value_counts()


# In[15]:


dmdata['Gender'].value_counts()


# In[16]:


dmdata['Initial_admin'].value_counts()


# In[17]:


dmdata['Initial_admin'].value_counts()


# In[18]:


dmdata['Complication_risk'].value_counts()


# In[19]:


dmdata['Services'].value_counts()


# In[20]:


# List of columns for ordinal encoding
columns_to_encode = [ 'Interaction', 'UID', 'Customer_id', 'Complication_risk']

# Apply label encoding to each column in the list
for column in columns_to_encode:
    dmdata[column] = label_encoder.fit_transform(dmdata[column])
dmdata[columns_to_encode]


# In[21]:


categorical_columns = dmdata.select_dtypes(include=['object']).columns.tolist()
("Remaining Categorical columns identified:", categorical_columns)


# In[22]:


# One-hot encoding for nominal categorical variables with more than two levels
dmdata = pd.get_dummies(dmdata,columns=['State', 'City', 'County', 'Area', 'TimeZone', 'Job','Marital', 'Gender', 'Initial_admin', 'Services'], drop_first=False,dtype=int)


# In[23]:


dmdata.info()


# In[24]:


#Check the distribution of target feature
dmdata['ReAdmis'].value_counts()


# In[25]:


#Feature selection to reduce the dimensionality, based on statistical tests using selectKBest

from sklearn.feature_selection import SelectKBest, f_classif

# Separate features and target
X = dmdata.drop(columns=['ReAdmis'])
y = dmdata['ReAdmis']

# f_classif is used to compute the ANOVA F-value for each feature
# k-'all'  indicates that all features will be scored, but no features are removed

selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Get the scores and p-values for each feature
scores = selector.scores_
p_values = selector.pvalues_

# Create a data frame to display feature scores and p-values
feature_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'Score': scores,
    'P-Value': p_values
}).sort_values(by='P-Value')

# Filter the DataFrame for features with p-value < 0.05
filtered_df = feature_scores_df[feature_scores_df['P-Value'] < 0.05]
print(filtered_df)


# In[26]:


filtered_df = feature_scores_df[(feature_scores_df['P-Value'] <= 0.0)]
(filtered_df.head(50))


# In[27]:


filtered_df = feature_scores_df[(feature_scores_df['P-Value'] > 0.0) & (feature_scores_df['P-Value'] <= 0.02)]
print(filtered_df.head(50))
filtered_df = feature_scores_df[(feature_scores_df['P-Value'] > 0.02) & (feature_scores_df['P-Value'] <= 0.04)]
print(filtered_df.head(50))
filtered_df = feature_scores_df[(feature_scores_df['P-Value'] <= 0.05) & (feature_scores_df['P-Value'] >= 0.04)]
print(filtered_df.head(50))
filtered_df = feature_scores_df[(feature_scores_df['P-Value'] > 0.05) & (feature_scores_df['P-Value'] < 0.07) ]
print(filtered_df.head(50))
filtered_df = feature_scores_df[(feature_scores_df['P-Value'] > 0.07) & (feature_scores_df['P-Value'] < 0.09) ]
print(filtered_df.head(50))
filtered_df = feature_scores_df[(feature_scores_df['P-Value'] > 0.09) & (feature_scores_df['P-Value'] < 1) ]
print(filtered_df.head(50))


# In[28]:


features_to_select = [
    'TotalCharge', 'Initial_days', 'Services_Intravenous', 'Population','Marital_Divorced',
    'Initial_admin_EmergencyAdmission', 'Children', 'Services_CTScan',
    'Asthma',  'ReAdmis'
]
#selecing feature 'Asthma' from the list based on practical significance
#Avoiding encoded categorical variables since they may cause overfitting
knndata = dmdata[features_to_select]
print(knndata.head())


# In[29]:


#Features selected for analysis
knndata.info()


# In[30]:


# Outliers in selected continuous variables by visualizing
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=knndata['TotalCharge'], color="steelblue")
plt.title('Boxplot of TotalCharge')

plt.subplot(1, 2, 2)
sns.boxplot(y=knndata['Initial_days'], color="steelblue")
plt.title('Boxplot of Initial_days')

plt.tight_layout()
plt.show()


# In[31]:


knndata.describe()


# In[32]:


#K-NN uses distance when making predictions, features need to be on similar a scale
#Normalize  the data using StandardScaler

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
dm_cont=knndata.drop(['ReAdmis','Initial_admin_EmergencyAdmission','Services_CTScan','Asthma','Services_Intravenous','Marital_Divorced'],axis=1)
dm_respstd=pd.DataFrame(scaler.fit_transform(dm_cont),columns=dm_cont.columns)
dm_resp=knndata[['ReAdmis','Initial_admin_EmergencyAdmission','Services_CTScan','Asthma','Services_Intravenous','Marital_Divorced']]
frames=[dm_resp,dm_respstd]
dmdata_std=pd.concat(frames,axis=1)
dmdata_std.head()


# In[33]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate VIF (variance Inflation Factor) for each selected feature
vif_stddata = pd.DataFrame()
vif_stddata["Feature"] = dmdata_std.columns
vif_stddata["VIF"] = [variance_inflation_factor(dmdata_std.values, i) for i in range(dmdata_std.shape[1])]

print(vif_stddata)

#Note: KNN is less sensitive to multicollinearity compared to linear models,VIF scores don’t directly affect its predictions.


# In[34]:


dmdata_std.info()


# In[35]:


import warnings
warnings.filterwarnings('ignore')
# Plot histograms for all features
knndata.drop('ReAdmis', axis=1).hist(figsize=[15, 15],color='olivedrab')
plt.suptitle("Distribution before Scaling", fontsize=16)
plt.show()


# In[36]:


# Plot histograms for all features after scaling
dmdata_std.drop('ReAdmis', axis=1).hist(figsize=[15, 15],color='darkturquoise')
plt.suptitle("Distribution after Scaling", fontsize=16)
plt.show()


# In[37]:


#Export  preprocessed data
dmdata_std.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/D209preprocesseddata.csv",index=False)


# # Split the data into train and test

# In[38]:


from sklearn.model_selection import train_test_split

#response and predictors
X=dmdata_std.drop(['ReAdmis'],axis=1)
y=dmdata_std['ReAdmis']

#splitting training and test data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)

# 0.3 means that 30% of the data will be used for testing
#  random_state=21 ensures that eeach time we run the code, the split will be the same
#  stratify=y ensures that the training and testing sets have the same proportion of each class as the original dataset


frames_train=[X_train,y_train]
dm_train=pd.concat(frames_train,axis=1)

frames_test=[X_test,y_test]
dm_test=pd.concat(frames_test,axis=1)


dm_train.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/traindata.csv",index=False)
dm_test.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/testdata.csv",index=False)


# In[39]:


dm_train = dm_train.iloc[:, 1:]
print(dm_train.head())


# In[40]:


#confirm stratified split
# Confirm split sizes in training
print("Training set size:", len(dm_train))
# Confirm split sizes in test
print("Testing set size:", len(dm_test))
# Check value counts for 'ReAdmis' in training set
print("Training set 'ReAdmis' distribution:\n", dm_train['ReAdmis'].value_counts())
# Check value counts for 'ReAdmis' in test set
print("Testing set 'ReAdmis' distribution:\n", dm_test['ReAdmis'].value_counts())
# Calculate and print the proportion of 'ReAdmis' == 0 in both sets
train_proportion = len(dm_train[dm_train['ReAdmis'] == 0]) / len(dm_train)
test_proportion = len(dm_test[dm_test['ReAdmis'] == 0]) / len(dm_test)
print("Proportion of 'ReAdmis' == 0 in training set:", train_proportion)
print("Proportion of 'ReAdmis' == 0 in testing set:", test_proportion)



# # Initial model

# In[41]:


# Build the initial K-NN model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#initial KNN classification
knn=KNeighborsClassifier(n_neighbors=10)   #classifier will consider the 10 nearest neighbors when making predictions
knn.fit(X_train,y_train)                   #Trains the KNN classifier on the feature values and labels

print("\nTest accuracy score:")
print(knn.score(X_test,y_test))            # to measure how correctly the model classify the test data
print("\nTrain accuracy score:")
print(knn.score(X_train,y_train))          ## to measure how correctly the model classify the training data

y_pred=knn.predict(X_test)  #predict the labels for the test data



# In[42]:


#initial metrics to determine the model accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(matrix)
print("\nClassification Report:")
print(classification_report(y_test,y_pred))



# In[43]:


#Model accuracy, specificity,precision and sensitivity
from sklearn.metrics import  recall_score

tp=matrix[1,1]  #true positives
tn=matrix[0,0]  #true negatives
fn=matrix[0,1]  #false engatives
fp=matrix[1,0]  #false positives

Accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy:", Accuracy)

Precision = tp / ( tp + fp )
print("Precision:", Precision)

sensitivity = recall_score(y_test, y_pred)
print("Sensitivity", sensitivity )

specificity = tn / (tn + fp)
print("specificity:", specificity)



# # ROC

# In[44]:


#ROC (Receiver Operating Characteristic) curve for initial model to evaluate the performance of classifier
from sklearn.metrics import roc_curve
y_pred_prob=knn.predict_proba(X_test)[:,1]   #predict the probabilities of the test_data for each class,selects the probabilities of +ve class
fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)   #plotting the TPR against the FPR at various threshold settings.

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='KNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('KNN ROC Curve')
plt.show()

#Note:The closer the curve follows the top-left corner (high TPR and low FPR), the better the model's performance.


# In[45]:


#finding the Area Under the ROC Curve (AUC) for initial classification model
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_test,y_pred_prob)
print('AUC: {}'.format(auc))

#Note:
# AUC of 0.5 suggests that the model performs no better than random chance
#AUC closer to 1 indicates excellent model performance
#AUC closer to 0 indicates a very poor model


# # Model complexity curve

# In[46]:


# To observe how model complexity influences accuracy and to detect potential overfitting or underfitting

# Initialize dictionaries to store accuracies
train_accuracies = {}
test_accuracies = {}

# Range of neighbors to test
neighbors = np.arange(1, 30) #Allows to see the overall trend in model performance across a broader spectrum of k values

# Loop through different values of n_neighbors
for neighbor in neighbors:
    # Initialize the KNN classifier with the current number of neighbors
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    
    # Fit the KNN model trains the KNN classifier using the training data
    knn.fit(X_train, y_train)
    
    # Record training and testing accuracies
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy", marker='o')
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy", marker='o')
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


# # Cross Validation

# In[59]:


#find the optimized model

from sklearn.model_selection import GridSearchCV  #to perform an exhaustive search over specified hyperparameter values

param_grid = {
    'n_neighbors': np.arange(5, 155,5),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']  #specifies the parameters for KNN before fitting
    
}
knn = KNeighborsClassifier()
knn_gridcv=GridSearchCV(knn,param_grid,cv=10)  #with cv=10 dataset is split into 10 parts,and model is trained and validated 10 times

knn_gridcv.fit(X_train,y_train)  #Fits the model using the training data and tries all combinations of the specified hyperparameters.

print('Best parameters: {}'.format(knn_gridcv.best_params_))
print('Best score: {}'.format(knn_gridcv.best_score_))


# In[62]:


#ROC-AUC of optimized model
y_pred_prob=knn_gridcv.predict_proba(X_test)[:,1]   #predict the probabilities of the test_data for each class,selects the probabilities of +ve class
fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)   #plotting the TPR against the FPR at various threshold settings.

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='KNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('KNN ROC Curve')
plt.show()

auc=roc_auc_score(y_test,y_pred_prob)
print('AUC: {}'.format(auc))


# # KNN using the modified hyperparameters to check consistency 

# In[71]:


# slightly different data splits, to tets the consistancy of model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56,stratify=y)


knn_optimized=KNeighborsClassifier(metric='manhattan',n_neighbors=10,weights='distance')

knn_optimized.fit(X_train,y_train) #trains the KNN classifier using the training data
print("\nTest accuracy score:")
print(knn_optimized.score(X_test,y_test))            # to measure how correctly the model classify the test data
print("\nTrain accuracy score:")
print(knn_optimized.score(X_train,y_train))          ## to measure how correctly the model classify the training data

y_pred_opt=knn_optimized.predict(X_test)  #predict the labels for the test data


#metrics
matrix2=confusion_matrix(y_test,y_pred_opt)
print("\n",matrix2)
print("\n", classification_report(y_test,y_pred_opt))


# In[72]:


from sklearn.metrics import accuracy_score
accuracy_model = accuracy_score(y_test, y_pred_opt)
print("Accuracy of the model",accuracy_model )


# In[73]:


tp=matrix2[1,1]  #true positives
tn=matrix2[0,0]  #true negatives
fn=matrix2[0,1]  #false engatives
fp=matrix2[1,0]  #false positives


Accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy:", Accuracy)


Precision = tp / ( tp + fp )
print("Precision:", Precision)

sensitivity = recall_score(y_test, y_pred_opt)
print("Sensitivity", sensitivity )

specificity = tn / (tn + fp)
print("specificity:", specificity)


# In[74]:


import numpy as np

# Count the number of instances for each class in the actual and predicted outcomes
actual_distribution = np.bincount(y_test)
predicted_distribution = np.bincount(y_pred_opt)

print(f"Actual distribution of 'ReAdmis': {actual_distribution}")
print(f"Predicted distribution of ReAdmis: {predicted_distribution}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




