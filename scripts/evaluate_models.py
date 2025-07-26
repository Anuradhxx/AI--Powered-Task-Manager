import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
import json
from datetime import datetime

# -------------------
# File Paths
# -------------------
csv_file_path = "output_tfidf_features.csv"
vectorizer_path = "features/features_tfidf.pkl"
nb_model_path = "models/task_classifier_nb.pkl"

# Create results directory
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# -------------------
# Load Data
# -------------------
print("[INFO] Loading CSV data...")
try:
    df = pd.read_csv(csv_file_path)
    print(f"[INFO] Loaded {len(df)} rows from CSV")
except FileNotFoundError:
    print(f"[ERROR] File {csv_file_path} not found!")
    exit(1)

# Check required columns
if 'Project Type' not in df.columns:
    print(f"[ERROR] Available columns: {list(df.columns)}")
    raise KeyError("Column 'Project Type' not found in CSV.")
if 'Project Name' not in df.columns or 'Task Name' not in df.columns:
    print(f"[ERROR] Available columns: {list(df.columns)}")
    raise KeyError("Columns 'Project Name' and/or 'Task Name' not found in CSV.")

# Combine text fields and clean data
print("[INFO] Preparing text data...")
df['Project Name'] = df['Project Name'].fillna('').astype(str)
df['Task Name'] = df['Task Name'].fillna('').astype(str)
df['combined_text'] = df['Project Name'] + " " + df['Task Name']

# Remove empty texts
df = df[df['combined_text'].str.strip() != '']
combined_texts = df['combined_text'].tolist()
y = df['Project Type']

print(f"[INFO] Prepared {len(combined_texts)} text samples")
print(f"[INFO] Target classes: {y.value_counts().to_dict()}")

# -------------------
# Load TF-IDF Vectorizer
# -------------------
print("[INFO] Loading TF-IDF vectorizer...")
try:
    loaded_data = joblib.load(vectorizer_path)
    
    # Handle different save formats
    if isinstance(loaded_data, tuple):
        print("[INFO] Loaded tuple format - extracting vectorizer...")
        if len(loaded_data) == 2:
            # Format: (X_features, vectorizer)
            vectorizer = loaded_data[1]
        else:
            # Try to find TfidfVectorizer in tuple
            vectorizer = None
            for item in loaded_data:
                if hasattr(item, 'transform') and hasattr(item, 'vocabulary_'):
                    vectorizer = item
                    break
            if vectorizer is None:
                raise ValueError("Could not find TfidfVectorizer in loaded tuple")
    elif hasattr(loaded_data, 'transform') and hasattr(loaded_data, 'vocabulary_'):
        # Direct vectorizer object
        vectorizer = loaded_data
    else:
        print("[ERROR] Unknown vectorizer format. Creating new vectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        print("[INFO] Fitting new vectorizer on current data...")
        vectorizer.fit(combined_texts)
        # Save the new vectorizer
        os.makedirs("features", exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)
        
except FileNotFoundError:
    print("[WARNING] Vectorizer file not found. Creating new vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    print("[INFO] Fitting new vectorizer...")
    vectorizer.fit(combined_texts)
    # Save the new vectorizer
    os.makedirs("features", exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
except Exception as e:
    print(f"[ERROR] Error loading vectorizer: {e}")
    print("[INFO] Creating new vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    vectorizer.fit(combined_texts)
    os.makedirs("features", exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)

# Transform text data
print("[INFO] Transforming text data...")
try:
    X = vectorizer.transform(combined_texts)
    print(f"[INFO] Feature matrix shape: {X.shape}")
except Exception as e:
    print(f"[ERROR] Error transforming text: {e}")
    print("[INFO] Re-fitting vectorizer...")
    vectorizer.fit(combined_texts)
    X = vectorizer.transform(combined_texts)
    print(f"[INFO] Feature matrix shape: {X.shape}")

# Check if we have enough samples for train-test split
if len(set(y)) < 2:
    print("[ERROR] Need at least 2 different classes for classification")
    exit(1)

min_class_count = y.value_counts().min()
if min_class_count < 2:
    print(f"[WARNING] Some classes have very few samples: {y.value_counts().to_dict()}")
    stratify_param = None  
else:
    stratify_param = y

# -------------------
# Train-Test Split
# -------------------
print("[INFO] Splitting data...")
test_size = min(0.3, max(0.1, min_class_count / len(y))) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify_param
)

print(f"[INFO] Train set: {X_train.shape[0]} samples")
print(f"[INFO] Test set: {X_test.shape[0]} samples")

# -------------------
# Initialize Results Storage
# -------------------
evaluation_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'total_samples': len(df),
        'features': X.shape[1],
        'classes': y.value_counts().to_dict(),
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0]
    },
    'models': {}
}

# -------------------
# Load and Evaluate Naive Bayes Model
# -------------------
try:
    print("[INFO] Loading Naive Bayes model...")
    nb_model = joblib.load(nb_model_path)
    nb_pred = nb_model.predict(X_test)
    nb_available = True
except FileNotFoundError:
    print("[WARNING] Naive Bayes model not found. Skipping NB evaluation.")
    nb_available = False
except Exception as e:
    print(f"[WARNING] Error loading NB model: {e}. Skipping NB evaluation.")
    nb_available = False

# -------------------
# Train SVM Model
# -------------------
print("[INFO] Training SVM model...")
try:
    svm_model = SVC(kernel='linear', random_state=42, C=1.0)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    print("[INFO] SVM training completed")
except Exception as e:
    print(f"[ERROR] Error training SVM: {e}")
    exit(1)

# -------------------
# Enhanced Evaluation Function with Result Storage
# -------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"{name} Model Evaluation")
    print(f"{'='*50}")
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nClassification Report:")
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        
        # Store results for saving
        model_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict,
            'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
            'true_labels': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
        }
        
        evaluation_results['models'][name] = model_results
        
        return accuracy, precision, recall
    except Exception as e:
        print(f"[ERROR] Error evaluating {name}: {e}")
        return 0, 0, 0

# -------------------
# Final Evaluation
# -------------------
results = {}

if nb_available:
    nb_acc, nb_prec, nb_rec = evaluate_model("Naive Bayes", y_test, nb_pred)
    results['Naive Bayes'] = {'Accuracy': nb_acc, 'Precision': nb_prec, 'Recall': nb_rec}

svm_acc, svm_prec, svm_rec = evaluate_model("SVM", y_test, svm_pred)
results['SVM'] = {'Accuracy': svm_acc, 'Precision': svm_prec, 'Recall': svm_rec}

# -------------------
# Model Comparison (if both models available)
# -------------------
if len(results) >= 2:
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f}")
    
    # Find best performing model
    best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
    print(f"\n🏆 Best Performing Model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.4f})")
    
    # Performance difference
    if 'Naive Bayes' in results and 'SVM' in results:
        nb_acc = results['Naive Bayes']['Accuracy']
        svm_acc = results['SVM']['Accuracy']
        diff = abs(nb_acc - svm_acc)
        better_model = 'SVM' if svm_acc > nb_acc else 'Naive Bayes'
        print(f"📊 Performance Difference: {diff:.4f} ({diff*100:.2f}%)")
        print(f"✅ {better_model} performs better")

# -------------------
# Save Results to Files
# -------------------
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# 1. Save detailed JSON results
json_filename = f"{results_dir}/evaluation_results_{timestamp_str}.json"
with open(json_filename, 'w') as f:
    json.dump(evaluation_results, f, indent=2)
print(f"\n[INFO] Detailed results saved to: {json_filename}")

# 2. Save summary CSV for easy comparison
summary_data = []
for model_name, metrics in results.items():
    summary_data.append({
        'Model': model_name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'Timestamp': evaluation_results['timestamp']
    })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    csv_filename = f"{results_dir}/model_comparison_{timestamp_str}.csv"
    summary_df.to_csv(csv_filename, index=False)
    print(f"[INFO] Summary CSV saved to: {csv_filename}")

# 3. Save detailed text report
report_filename = f"{results_dir}/evaluation_report_{timestamp_str}.txt"
with open(report_filename, 'w') as f:
    f.write("="*60 + "\n")
    f.write("MODEL EVALUATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(f"Timestamp: {evaluation_results['timestamp']}\n")
    f.write(f"Dataset: {evaluation_results['dataset_info']['total_samples']} samples\n")
    f.write(f"Features: {evaluation_results['dataset_info']['features']} TF-IDF features\n")
    f.write(f"Classes: {evaluation_results['dataset_info']['classes']}\n")
    f.write(f"Train Set: {evaluation_results['dataset_info']['train_samples']} samples\n")
    f.write(f"Test Set: {evaluation_results['dataset_info']['test_samples']} samples\n\n")
    
    for model_name, metrics in results.items():
        f.write(f"{model_name} Results:\n")
        f.write(f"  Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)\n")
        f.write(f"  Precision: {metrics['Precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['Recall']:.4f}\n\n")
        
        # Add confusion matrix from stored results
        if model_name in evaluation_results['models']:
            cm = evaluation_results['models'][model_name]['confusion_matrix']
            f.write(f"  Confusion Matrix:\n")
            for row in cm:
                f.write(f"    {row}\n")
            f.write("\n")
    
    if len(results) >= 2:
        f.write("Model Comparison:\n")
        best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
        f.write(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.4f})\n")

print(f"[INFO] Text report saved to: {report_filename}")

# -------------------
# Save SVM model
# -------------------
try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(svm_model, "models/task_classifier_svm.pkl")
    print(f"\n[INFO] SVM model saved as 'models/task_classifier_svm.pkl'")
except Exception as e:
    print(f"[ERROR] Error saving SVM model: {e}")

# -------------------
# Summary
# -------------------
print(f"\n{'='*50}")
print("EVALUATION SUMMARY")
print(f"{'='*50}")
print(f"Models Evaluated: {len(results)}")
print(f"Dataset: {len(df)} samples, {X.shape[1]} features")
print(f"Test Set: {len(y_test)} samples")

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

print(f"\n[INFO] All results saved in '{results_dir}' directory:")
print(f"  📊 JSON: {json_filename}")
if 'csv_filename' in locals():
    print(f"  📈 CSV: {csv_filename}")
print(f"  📄 Report: {report_filename}")

print("\n[INFO] Evaluation completed successfully!")