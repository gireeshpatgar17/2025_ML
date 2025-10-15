#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


df=pd.read_csv("emails.csv");


# In[7]:


df.head()


# In[41]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from math import log, pi

# Step 1: Load dataset
data = pd.read_csv("emails.csv")

# Step 2: Keep only numeric columns
data = data.select_dtypes(include=['number'])

# Step 3: Separate features and target
X = data.drop(columns=['Prediction'])
y = data['Prediction']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Compute priors
classes = np.unique(y_train)
priors = {c: np.mean(y_train == c) for c in classes}
print("Prior Probabilties are:")
print(priors)


# In[37]:


# Step 8: Predict classes and store posteriors
y_pred = []
posterior_probs = []  # store posterior probabilities

for i in range(len(X_test)):
    posteriors = {c: compute_posterior(X_test.iloc[i], c) for c in classes}
    
    # Convert log-posteriors to normal probabilities
    max_log = max(posteriors.values())
    probs = {c: np.exp(posteriors[c] - max_log) for c in posteriors}  # prevent overflow
    total = sum(probs.values())
    probs = {c: probs[c]/total for c in probs}  # normalize to sum=1
    
    posterior_probs.append(probs)  # store probabilities
    
    predicted_class = max(posteriors, key=posteriors.get)
    y_pred.append(predicted_class)

# Step 9: Evaluate performance
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", round(accuracy, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Print posterior probabilities for first 5 test samples
for i in range(10):
    print(f"Email {i+1} posterior probabilities: {posterior_probs[i]}")


# In[ ]:




