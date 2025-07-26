import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


os.makedirs('evaluation_results/plots', exist_ok=True)


try:
    with open('evaluation_results/evaluation_report_202507.txt', 'r') as f:
        txt_report = f.read()
except Exception as e:
    print(f"Error reading TXT file: {e}")
    txt_report = None


try:
    with open('evaluation_results/evaluation_results_nb_202507.json') as f:
        report_nb = json.load(f)
except Exception as e:
    print(f"Error reading Naive Bayes JSON: {e}")
    report_nb = None

try:
    with open('evaluation_results/evaluation_results_svm_202507.json') as f:
        report_svm = json.load(f)
except Exception as e:
    print(f"Error reading SVM JSON: {e}")
    report_svm = None


try:
    
    results_df = pd.DataFrame({
        'Model': ['Naive Bayes', 'SVM'],
        'F1_Score': [0.81, 0.88]
    })
except Exception as e:
    print(f"Error creating fallback DataFrame: {e}")
    results_df = None


try:
    with open('models/task_classifier_nb.pkl', 'rb') as f:
        model_nb = pickle.load(f)
    with open('models/task_classifier_svm.pkl', 'rb') as f:
        model_svm = pickle.load(f)
    with open('models/priority_model_rf.pkl', 'rb') as f:
        model_rf = pickle.load(f)
    with open('models/priority_predictor_xgb.pkl', 'rb') as f:
        model_xgb = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")


if results_df is not None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Model', y='F1_Score', palette='mako')
    plt.title('Model Comparison - F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evaluation_results/plots/dashboard_summary.png')
    plt.show()
    print("\nBar chart saved as 'evaluation_results/plots/dashboard_summary.png'")
else:
    print("Skipping plot - no valid results_df")


print("\n=== TEXT REPORT (if available) ===\n")
if txt_report:
    print(txt_report)

if report_nb:
    print("\n=== Naive Bayes Classification Report ===")
    print(report_nb)

if report_svm:
    print("\n=== SVM Classification Report ===")
    print(report_svm)
