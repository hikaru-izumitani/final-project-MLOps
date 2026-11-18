# mlflow server --port 8080 --backend-store-uri sqlite:///mlruns.db
# mlflow quickstart https://mlflow.org/docs/latest/ml/getting-started/quickstart/
# iris dataset https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
#import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("classification_experiment")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

with mlflow.start_run():
    # just train the model normally
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    print("Training completed and autologged successfully!")



