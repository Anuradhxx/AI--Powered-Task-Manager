import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report


os.makedirs('models', exist_ok=True)


features = pd.read_csv("output_tfidf_features.csv")  
features = features.select_dtypes(include=['number'])  

data = pd.read_csv("Cleaned_data.csv")
labels = data["Priority"]  


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)


nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb, zero_division=1))


with open("models/task_classifier_nb.pkl", "wb") as f:
    pickle.dump(nb_model, f)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, zero_division=1))




