import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


# ===============================
# Load dataset preprocessing
# ===============================
data_path = "PredDeposit_preprocessed.csv"
df = pd.read_csv(data_path)

# Tentukan kolom target (samakan dengan notebook)
target_col = "deposit"   

X = df.drop(columns=[target_col])
y = df[target_col]

# ===============================
# Handle missing value pada TARGET
# ===============================
y = y.fillna(y.mode()[0])


# ===============================
# Split data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ===============================
# MLflow setup (LOCAL)
# ===============================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Deposit-Prediction-Basic")

mlflow.sklearn.autolog()

# ===============================
# Training model
# ===============================
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
