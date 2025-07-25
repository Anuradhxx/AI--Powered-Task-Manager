import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os


with open('features/features_tfidf.pkl', 'rb') as f:
    X, y = pickle.load(f)


df = pd.read_csv('Cleaned_data.csv')
y = df['Priority']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)
with open('models/priority_model_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_train_enc, y_test_enc = train_test_split(y_encoded, test_size=0.2, random_state=42)


xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(X_train, y_train_enc)


y_pred_xgb = xgb.predict(X_test)
y_pred_xgb_labels = label_encoder.inverse_transform(y_pred_xgb)
y_test_labels = label_encoder.inverse_transform(y_test_enc)

print("\nXGBoost:\n", classification_report(y_test_labels, y_pred_xgb_labels))

# Save model
with open('models/priority_model_xgb.pkl', 'wb') as f:
    pickle.dump(xgb, f)
