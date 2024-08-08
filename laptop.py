#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[2]:


pip install streamlit


# In[4]:


import streamlit as st


# In[49]:


st.title('Laptop Price Prediction using Linear Regression')
data = pd.read_csv(r"C:\Users\pc\Desktop\laptop_data.csv")
data


# In[67]:


data.columns


# In[78]:


#data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)
numerical_features = ['Weight', 'Inches']
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
numerical_features


# In[79]:


categorical_features = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
categorical_features


# In[76]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
preprocessor


# In[80]:


pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])
pipeline


# In[84]:


X = data.drop(columns='Price')
y = data['Price']


# In[85]:


X


# In[86]:


y


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[88]:


X_train


# In[89]:


X_test


# In[90]:


y_train


# In[91]:


y_test


# In[61]:


pipeline.fit(X_train, y_train)


# In[101]:


#from sklearn.linear_model import LinearRegression
#lin_reg = LinearRegression()
#lin_reg.fit(X_train,y_train)
#y_pred = lin_reg.predict(X_test)
y_pred = pipeline.predict(X_test)


# In[95]:


mse = mean_squared_error(y_test, y_pred)
f'Mean Squared Error: {mse}'


# In[116]:


r2 = r2_score(y_test, y_pred)
f'R^2 Score: {r2}'


# In[5]:


st.write(f'Mean Squared Error: {mse}')
st.write(f'R^2 Score: {r2}')


# In[117]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[6]:


streamlit run app.py

