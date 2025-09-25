import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

#load data =====
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.xls")

X = df.drop("Churn", axis=1)
y = df["Churn"].astype(str).str.strip().str.capitalize().map({"No": 0, "Yes": 1})

if "TotalCharges" in X.columns:
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

#pisahin kolom num & cat
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns


#preprocessing =====
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

#models =====
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

#train-Test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

best_model = None
best_score = -np.inf
best_name = None



#training & evaluation ======
for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    
    score = roc_auc_score(y_test, y_proba)
    if score > best_score:
        best_score = score
        best_model = clf
        best_name = name


#save best model ======
os.makedirs("models", exist_ok=True)
print(f"\nBest model: {best_name} (ROC-AUC = {best_score:.4f})")
joblib.dump(best_model, f"models/best_model_{best_name}.pkl")


#feature importance(LogReg) =====
if best_name == "LogReg":
    coefs = best_model.named_steps["classifier"].coef_[0]
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 important features (LogReg):")
    for feat, val in importance[:10]:
        print(f"{feat}: {val:.4f}")

