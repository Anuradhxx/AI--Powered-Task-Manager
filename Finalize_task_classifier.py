import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

X = pd.read_csv("output_tfidf_features.csv")      
df = pd.read_csv("data/tasks.csv")
y = df["Priority"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])


svm_model = SVC()
svm_model.fit(X_train, y_train.values.ravel())


joblib.dump(svm_model, "models/task_classifier_svm.pkl")
print("SVM model for Task Classification saved.")
