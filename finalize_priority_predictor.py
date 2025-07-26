import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder


X = pd.read_csv("output_tfidf_features.csv")
X = X.select_dtypes(include=['number'])  


y = pd.read_csv("Cleaned_data.csv")
labels = y['Priority'] 


X = X[~labels.isna()]
labels = labels[~labels.isna()]


le = LabelEncoder()
labels_encoded = le.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(X, labels_encoded, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)  


joblib.dump(xgb_model, "models/priority_predictor_xgb.pkl")
print("XGBoost model for Priority Prediction saved.")
