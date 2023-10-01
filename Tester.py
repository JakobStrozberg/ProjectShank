#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:47:35 2023

@author: jakobstrozberg
"""
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np
df = pd.read_csv('Project1Data.csv')
X = df[['X', 'Y', 'Z']]

# Load the model
loaded_model = load('random_forest_model.joblib')

# New data
new_data = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])

# Standardize the new data just like the training data
scaler = StandardScaler().fit(X)  # Assuming 'X' is available; replace with appropriate load statement if not.
new_data = scaler.transform(new_data)

# Make predictions
predictions = loaded_model.predict(new_data)

# Print predictions
print("Predicted steps: ", predictions)