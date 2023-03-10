#Importing necessary packages

import numpy as np
import pandas as pd


# In[2]:


#Reading the data

loan_data = pd.read_csv("D:\AIML\Dataset\loan.csv")


# In[3]:


#Dropping the irrelevant column

loan_data.drop('Loan_ID',axis=1,inplace=True)


# In[4]:


#Printing the unique values of 'Object' columns

print(loan_data['Gender'].unique())
print(loan_data['Education'].unique())
print(loan_data['Married'].unique())
print(loan_data['Dependents'].unique())
print(loan_data['Self_Employed'].unique())
print(loan_data['Property_Area'].unique())
print(loan_data['Loan_Status'].unique())


# In[5]:


#Selecting the 'Object' columns

loan_data.select_dtypes(include='object').columns


# In[6]:


#Label encoding the 'Object' columns

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

feat_array= ohe.fit_transform(loan_data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Property_Area']]).toarray() #Values of the new 'Object' columns


# In[7]:


#Getting the new column names of the 'Object' columns 

feat_names = ohe.get_feature_names_out()


# In[8]:


#Creating a dataframe of the above label encoded data

ohe_df = pd.DataFrame(feat_array,columns = feat_names)


# In[9]:


#Concatenating the main dataset & label encoded dataset

df = pd.concat([loan_data,ohe_df],axis=1)


# In[10]:


#Selecting the 'Object' columns

df.select_dtypes(include='object').columns


# In[11]:


#Removing the object columns

df.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Property_Area'],axis=1,inplace=True)


# In[12]:


#Treating the null values of 'LoanAmount' with mean

def lo_am_mean(dataset,variable,value):
    dataset[variable+"_mean"] = dataset[variable].fillna(value)

Amount_mean = df.LoanAmount.mean()    


# In[13]:


#Calling the mean function

lo_am_mean(df,'LoanAmount',Amount_mean)


# In[14]:


#Treating the null values of 'LoanAmount' with mode

def lo_am_mode(dataset,variable,value):
    dataset[variable+"_mode"] = dataset[variable].fillna(value)
    
Amount_mode = df.LoanAmount.mode()


# In[15]:


#Calling the mode function

lo_am_mode(df,'LoanAmount',Amount_mode[0])


# In[16]:


#Treating the null values of 'LoanAmount' with median

def lo_am_median(dataset,variable,value):
    dataset[variable+"_median"] = dataset[variable].fillna(value)
    
Amount_median = df.LoanAmount.median()


# In[17]:


#Calling the mode function

lo_am_median(df,'LoanAmount',Amount_median)


# In[18]:


#Treating the null values of 'Loan_Amount_Term' with mean

def term_mean(dataset,variable,value):
    dataset[variable+"_mean"] = dataset[variable].fillna(value)
    
lo_term_mean = df.Loan_Amount_Term.mean()


# In[19]:


#calling the mean function

term_mean(df,'Loan_Amount_Term',lo_term_mean)


# In[20]:


#Treating the null values of 'Loan_Amount_Term' with mode

def term_mode(dataset,variable,value):
    dataset[variable+"_mode"] = dataset[variable].fillna(value)
    
lo_term_mode = df.Loan_Amount_Term.mode()


# In[21]:


#calling the mode function

term_mode(df,'Loan_Amount_Term',lo_term_mode[0])


# In[22]:


#Treating the null values of 'Loan_Amount_Term' with median

def term_median(dataset,variable,value):
    dataset[variable+"_median"] = dataset[variable].fillna(value)
    
lo_term_median = df.Loan_Amount_Term.median()


# In[23]:


#calling the median function

term_median(df,'Loan_Amount_Term',lo_term_median)


# In[24]:


#Treating the null values of 'Credit_History' with mean

def credit_mean(dataset,variable,value):
    dataset[variable+"_mean"] = dataset[variable].fillna(value)
    
cre_his_mean = df.Credit_History.mean()


# In[25]:


#Calling the mean function

credit_mean(df,'Credit_History',cre_his_mean)


# In[26]:


#Treating the null values of 'Credit_History' with mode

def credit_mode(dataset,variable,value):
    dataset[variable+"_mode"] = dataset[variable].fillna(value)
    
cre_his_mode = df.Credit_History.mode()


# In[27]:


#Calling the mode function

credit_mode(df,'Credit_History',cre_his_mode[0])


# In[28]:


#Treating the null values of 'Credit_History' with mode

def credit_median(dataset,variable,value):
    dataset[variable+"_median"] = dataset[variable].fillna(value)
    
cre_his_median = df.Credit_History.median()


# In[29]:


#Calling the median function

credit_median(df,'Credit_History',cre_his_median) 


# In[30]:


#Printing the standard deviation of original & new columns treated with null values

print(f"Standard deviation of original column {df['LoanAmount'].std()} Standard of mean column {df['LoanAmount_mean'].std()}")
print(f"Standard deviation of mode column {df['LoanAmount_mode'].std()} Standard of median column {df['LoanAmount_median'].std()}")

print(f"Standard deviation of original column {df['Loan_Amount_Term'].std()} Standard of mean column {df['Loan_Amount_Term_mean'].std()}")
print(f"Standard deviation of mode column {df['Loan_Amount_Term_mode'].std()} Standard of median column {df['Loan_Amount_Term_median'].std()}")

print(f"Standard deviation of original column {df['Credit_History'].std()} Standard of mean column {df['Credit_History_mean'].std()}")
print(f"Standard deviation of mode column {df['Credit_History_mode'].std()} Standard of median column {df['Credit_History_median'].std()}")


# In[31]:


#Dropping the irrelevant & null value columns

df.drop(['LoanAmount','Loan_Amount_Term','Credit_History','LoanAmount_mean','LoanAmount_median','Loan_Amount_Term_mean','Loan_Amount_Term_median','Credit_History_mean','Credit_History_median'],axis=1,inplace=True)


# In[32]:


#Declaring IV & DV

x = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']


# In[33]:


#Label encoding the DV

from sklearn.preprocessing import LabelEncoder

lo_stat = LabelEncoder()

lo_stat.fit(['Y','N'])

y = lo_stat.transform(y)


# In[34]:


#Splitting the dataset

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=150)

print("Train shape",train_x.shape,train_y.shape)
print("Test shape",test_x.shape,test_y.shape)


# In[35]:


# SVM Modeling

from sklearn import svm

loan_svm = svm.SVC(kernel='poly')

loan_svm.fit(train_x,train_y)


# In[36]:


#Evaluating the 'SVM' model using test data

svm_pred = loan_svm.predict(test_x)


# In[37]:


#Checking accuracy score

from sklearn.metrics import f1_score
print(f"Accuracy score = {f1_score(svm_pred,test_y,average='weighted')*100} %")


# In[ ]:




