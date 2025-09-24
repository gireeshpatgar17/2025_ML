#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


#

# Step 2: Load dataset
df = pd.read_csv("dummy_binary_classification.csv")  # make sure the file is in your notebook's working directory

# Step 3: Split into features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Step 4: Train-test split (2-way split: 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)  # increased iterations for convergence
log_reg.fit(X_train, y_train)

# Step 6: Predictions
y_pred = log_reg.predict(X_test)

# Step 7: Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





# In[5]:


# Step 1: Import libraries


# Step 2: Load dataset
df = pd.read_csv("dummy_binary_classification.csv")

# Step 3: Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 4: First split -> Train (60%) + Temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Step 5: Second split -> Validation (20%) + Test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Split sizes:")
print("Train:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test:", X_test.shape[0])

# Step 6: Train logistic regression on training set
log_reg = LogisticRegression(C=0.01, max_iter=1000)
log_reg.fit(X_train, y_train)

# Step 7: Evaluate on validation set
y_val_pred = log_reg.predict(X_val)
print("\n🔎 Validation Results")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Step 8: Final evaluation on test set
y_test_pred = log_reg.predict(X_test)
print("\n✅ Test Results")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))


# In[ ]:





# In[6]:


# Step 1: Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Step 2: Load dataset
df = pd.read_csv("dummy_binary_classification.csv")

# Step 3: Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 4: Initialize logistic regression
log_reg = LogisticRegression(max_iter=1000)

# Step 5: Define k-fold strategy (stratified to preserve class balance)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 6: Perform k-fold cross-validation
scores = cross_val_score(log_reg, X, y, cv=kf, scoring='accuracy')

# Step 7: Print results
print("✅ Accuracy for each fold:", scores)
print("✅ Mean Accuracy:", scores.mean())
print("✅ Standard Deviation:", scores.std())


# In[ ]:




