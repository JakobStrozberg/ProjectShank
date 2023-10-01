import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import joblib
import sklearn as sk



# Step 1, Creating a DataFrame from a dictionary
df = pd.read_csv('Project1Data.csv')
print(df.head())
#Extracting the data from the csv
x = df['X']
y = df['Y']
z = df['Z']
Step = df['Step']

#Step 2, Plotting the data
plt.xlabel('Data point values')
plt.ylabel('Number of data points')
plt.title('Project 1')
plt.plot(df)
plt.show()
#Analyzing and printing summary statistics
print(df.describe())

#Step 3, Finding and displaying correlation
correlation_matrix = df.corr()
print(correlation_matrix)
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



#step 4 Classification Model Development

# First import the necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Preparing train and test data, 'x', 'y' and 'z' are input, 'step' is output
X = df[['X', 'Y', 'Z']]
y = df['Step']

# Split the data in an 80%/20% train/test ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the feature scaling for optimized results
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define a function to fit and grid search a model
def get_best_model_and_accuracy(model, params, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)  # Instantiating GridSearchCV
    grid.fit(X_train, y_train)  # Fitting the model and the parameters to the GridSearchCV
    print(f"Best Accuracy: {grid.best_score_}")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Accuracy on test data: {grid.score(X_test, y_test)}")

# Define the models and parameters
models_and_parameters = {
    "logistic_regression": (LogisticRegression(max_iter=5000), 
                            {'C': [0.001, 0.01, 0.1, 1, 10]}),
    "svm": (SVC(), 
            {'C': [0.001, 0.01, 0.1, 1, 10]}),
    "random_forest": (RandomForestClassifier(), 
                      {'n_estimators': [10, 50, 100]})
}

# Train the models and display the best parameters
for model_name, (model, params) in models_and_parameters.items():
    print(f"==== Starting Grid Search for {model_name} ====")
    get_best_model_and_accuracy(model, params, X_train, y_train, X_test, y_test)
    
    
# The justifcation for the ML models: 
# The selection of Logistic Regression, Support Vector Machines (SVM), and Random Forest for classification 
# tasks is based on their versatility and efficiency in dealing with various prediction problems.
# **Logistic Regression**: It's a simple yet effective algorithm for binary and multiclass classification problems. 
# It's easy to implement, interpret and doesn't require high computation power.
# **Support Vector Machines (SVM)**: This is a powerful algorithm that can create complex decision boundaries, 
# even in cases of non-linearly separable data. It's effective in high dimensional spaces making it a good choice for many applications.
# **Random Forest**: This is a robust and versatile machine learning method (an ensemble of decision trees) 
# that can perform both regression and classification tasks. It has an excellent capability to handle overfitting and can 
# work well even with default parameter settings.

#Step 5 
#In general, if false positives and false negatives have different costs, it's better to look at precision and recall or the F1 score. If the costs are approximately the same and the classes are well balanced, then model accuracy can be used.
# Based on your results the Random Forest model has the highest accuracy on the test data, making it the best performer according to this metric.

from sklearn.metrics import classification_report

# Logistic Regression
log_reg = LogisticRegression(C=10, max_iter=5000)
log_reg.fit(X_train, y_train)

pred_lr = log_reg.predict(X_test)

print("Logistic Regression:\n", classification_report(y_test, pred_lr))

# SVM
svm_clf = SVC(C=1)
svm_clf.fit(X_train, y_train)

pred_svm = svm_clf.predict(X_test)

print("Support Vector Machine:\n", classification_report(y_test, pred_svm))

# Random Forest
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("Random Forest:\n", classification_report(y_test, pred_rf))

#Step 6
from joblib import dump

# Random Forest with best parameters from the Grid Search
final_model = RandomForestClassifier(n_estimators=10)
final_model.fit(X_train, y_train)

# Save the model as a joblib file
dump(final_model, 'random_forest_model.joblib')