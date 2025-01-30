from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


MODELS = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1, 10]
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'max_depth': [3, 5, 10],
        'criterion': ['gini', 'entropy']
    }),
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10]
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    'SVM': (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    'Naive Bayes': (GaussianNB(), {}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7]
    })
}


def evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    trained_models = {}
    best_params = {}
    for name, (model, params) in MODELS.items():
        if params:
            grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params[name] = grid_search.best_params_
        else:
            best_model = model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({'Model': name, 'Accuracy': accuracy, 'Best Parameters': best_params.get(name, 'N/A')})
        trained_models[name] = best_model
    return results, trained_models, best_params


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        column_examples = {
            col: df[col].dropna().unique().tolist()[:3]  
            for col in df.columns
        }
        return render_template('fields.html', columns=columns, filepath=filepath, column_examples=column_examples)


@app.route('/compare', methods=['POST'])
def compare():
    filepath = request.form['filepath']
    input_fields = request.form.getlist('input_fields')
    output_field = request.form['output_field']
    
    df = pd.read_csv(filepath)
    X = df[input_fields]
    y = df[output_field]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results, trained_models, best_params = evaluate_models(X_train, X_test, y_train, y_test)
    
    test_data = {field: float(request.form[field]) for field in input_fields}
    test_df = pd.DataFrame([test_data])
    test_df_scaled = scaler.transform(test_df)
    
    predictions = {}
    for model in trained_models:
        model_prediction = trained_models[model].predict(test_df_scaled)[0]
        predictions[model] = "Positive" if model_prediction == 1 else "Negative" if model_prediction == 0 else model_prediction
    
    return render_template('results.html', results=results, predictions=predictions, test_data=test_data, best_params=best_params)


if __name__ == '__main__':
    app.run(debug=True)
