# Tools
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = pd.read_csv('../input/diabetes_data_upload.csv')

# Preprocessing

def preprocess_inputs(df):
    df = df.copy()
    
    # Split X and y
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Binary encode X
    X = X.replace({'No': 0, 'Yes': 1})
    X = X.replace({'Female': 0, 'Male': 1})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
    
    # Scale X
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_inputs(data)

# Training

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "Support Vector Machine (RBF Kernel)": SVC(),
    "Neural Network": MLPClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()    
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Cross-validation

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "Support Vector Machine (RBF Kernel)": SVC(),
    "Neural Network": MLPClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()    
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Predict data by input
def predict(input_data) :
    input_data['Gender'] = 1 if input_data['Gender'] == 'Male' else 0
    input_data['Polyuria'] = 1 if input_data['Polyuria'] == 'Yes' else 0
    input_data['Polydipsia'] = 1 if input_data['Polydipsia'] == 'Yes' else 0
    input_data['sudden weight loss'] = 1 if input_data['sudden weight loss'] == 'Yes' else 0
    input_data['weakness'] = 1 if input_data['weakness'] == 'Yes' else 0
    input_data['Polyphagia'] = 1 if input_data['Polyphagia'] == 'Yes' else 0
    input_data['Genital thrush'] = 1 if input_data['Genital thrush'] == 'Yes' else 0
    input_data['visual blurring'] = 1 if input_data['visual blurring'] == 'Yes' else 0
    input_data['Itching'] = 1 if input_data['Itching'] == 'Yes' else 0
    input_data['Irritability'] = 1 if input_data['Irritability'] == 'Yes' else 0
    input_data['delayed healing'] = 1 if input_data['delayed healing'] == 'Yes' else 0
    input_data['partial paresis'] = 1 if input_data['partial paresis'] == 'Yes' else 0
    input_data['muscle stiffness'] = 1 if input_data['muscle stiffness'] == 'Yes' else 0
    input_data['Alopecia'] = 1 if input_data['Alopecia'] == 'Yes' else 0
    input_data['Obesity'] = 1 if input_data['Obesity'] == 'Yes' else 0
    # Lakukan hal yang sama untuk atribut lainnya

    input_df = pd.DataFrame([input_data])

    # Lakukan penskalaan pada data input menggunakan scaler yang telah dihasilkan sebelumnya
    scaled_input_df = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)

    all_predictions = {}
    for name, model in models.items():
        predictions = model.predict(scaled_input_df)[0]
        all_predictions[name] = predictions

    return all_predictions

predict_data = sys.argv[1]
predict_data = json.loads(predict_data)

predict_output = predict(predict_data)
print(json.dumps(predict_output))