#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn


# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_preprocessing_pipeline(data):
    #Identify numeric and categorical features
    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    #Handle missing values in numeric features
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

    #Detect and handle outliers in numeric features using IQR
    for feature in numeric_features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        data[feature] = np.where((data[feature] < lower_bound) | (data[feature] > upper_bound),
                                 data[feature].mean(), data[feature])

    #Normalize numeric features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_features])
    data[numeric_features] = scaler.transform(data[numeric_features])

    #Handle missing values in categorical features
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    return data


# In[3]:


data = pd.read_csv("MOCK_DATA.csv")

print("Original Data:")
print(data)


# In[4]:


#Perform data preprocessing
cleaned_data = data_preprocessing_pipeline(data)

print("Preprocessed Data:")
print(cleaned_data)


# In[6]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

def transform_data(data, n_components=0.95):
    # Identify numeric and categorical features
    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Apply One-Hot Encoding to categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cats = encoder.fit_transform(data[categorical_features])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))

    # Combine numeric and encoded categorical features
    transformed_data = pd.concat([data[numeric_features].reset_index(drop=True), encoded_cat_df], axis=1)

    # Apply PCA for dimensionality reduction (optional)
    pca = PCA(n_components=n_components)  # Retain 95% of variance
    transformed_data_pca = pca.fit_transform(transformed_data)

    return pd.DataFrame(transformed_data_pca)

# Perform data transformation
transformed_data = transform_data(cleaned_data)

# Print transformed data
print("Transformed Data:")
print(transformed_data.head())


# In[ ]:




