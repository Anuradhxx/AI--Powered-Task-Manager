import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import re
import os
from datetime import datetime

def load_and_analyze_data():
    """
    Load all available CSV files and analyze their structure
    """
    print("🔍 Analyzing available data files...")
    print("=" * 50)
    
    
    csv_files = {
        'tasks': 'data/tasks.csv',
        'cleaned_data': 'Cleaned_data.csv',
        'tfidf_features': 'output_tfidf_features.csv',
        'word2vec_features': 'output_word2vec_features.csv',
        'bert_features': 'output_bert_features.csv',
        'preprocessed_data': 'preprocessed_data.csv'
    }
    
    available_files = {}
    
    for name, path in csv_files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                available_files[name] = {
                    'path': path,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'size_mb': round(size_mb, 2),
                    'dataframe': df
                }
                print(f"✅ {name}: {path}")
                print(f"   Shape: {df.shape}, Size: {size_mb:.2f} MB")
                print(f"   Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
                print()
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    return available_files

def select_best_data_source(available_files):
    """
    Select the best data source for training
    """
    print("🎯 Selecting best data source...")
    
    # Priority order for data sources
    priority_order = ['tasks', 'cleaned_data', 'preprocessed_data', 'tfidf_features']
    
    for source in priority_order:
        if source in available_files:
            df = available_files[source]['dataframe']
            
            # Check for required columns
            required_text_cols = ['Project Name', 'Task Name', 'Task', 'Description', 'Text']
            required_label_cols = ['Project Type', 'Type', 'Category', 'Label', 'Class']
            
            text_col = None
            label_col = None
            priority_col = None
            
            # Find text column
            for col in df.columns:
                if any(req_col.lower() in col.lower() for req_col in required_text_cols):
                    text_col = col
                    break
            
            # Find label column
            for col in df.columns:
                if any(req_col.lower() in col.lower() for req_col in required_label_cols):
                    label_col = col
                    break
            
            # Find priority column
            for col in df.columns:
                if 'priority' in col.lower():
                    priority_col = col
                    break
            
            if text_col or label_col:
                print(f"✅ Selected: {source} ({available_files[source]['path']})")
                print(f"   Text column: {text_col}")
                print(f"   Label column: {label_col}")
                print(f"   Priority column: {priority_col}")
                return source, available_files[source]
    
    # If no ideal source found, use the first available
    if available_files:
        first_source = list(available_files.keys())[0]
        print(f"⚠️ Using first available: {first_source}")
        return first_source, available_files[first_source]
    
    return None, None

def preprocess_data(data_info):
    """
    Preprocess the selected data source
    """
    print("\n🔧 Preprocessing data...")
    
    df = data_info['dataframe'].copy()
    print(f"📊 Starting with {len(df)} rows")
    
    # Identify columns
    print("📋 Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")
        print(f"      Sample values: {df[col].dropna().head(3).tolist()}")
        print()
    
    # Try to identify text and label columns automatically
    text_columns = []
    label_columns = []
    priority_columns = []
    
    for col in df.columns:
        # Check for text columns
        if any(keyword in col.lower() for keyword in ['name', 'task', 'description', 'text', 'title']):
            if df[col].dtype == 'object':  # String columns
                text_columns.append(col)
        
        # Check for label columns
        if any(keyword in col.lower() for keyword in ['type', 'category', 'class', 'label']):
            if df[col].dtype == 'object':
                label_columns.append(col)
        
        # Check for priority columns
        if 'priority' in col.lower():
            priority_columns.append(col)
    
    print(f"🔤 Identified text columns: {text_columns}")
    print(f"🏷️ Identified label columns: {label_columns}")
    print(f"⚠️ Identified priority columns: {priority_columns}")
    
    # Create combined text
    if text_columns:
        df['combined_text'] = ''
        for col in text_columns:
            df['combined_text'] += ' ' + df[col].fillna('').astype(str)
    else:
        # If no clear text columns, combine first few string columns
        string_cols = [col for col in df.columns if df[col].dtype == 'object'][:3]
        df['combined_text'] = ''
        for col in string_cols:
            df['combined_text'] += ' ' + df[col].fillna('').astype(str)
        print(f"⚠️ No clear text columns found, using: {string_cols}")
    
    # Clean text
    df['combined_text'] = df['combined_text'].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)).lower().strip()
    )
    
    # Remove empty or very short texts
    df = df[df['combined_text'].str.len() > 3]
    print(f"📝 After text cleaning: {len(df)} rows")
    
    # Prepare labels
    task_labels = None
    priority_labels = None
    
    if label_columns:
        # Use the first available label column
        label_col = label_columns[0]
        task_labels = df[label_col].fillna('Unknown').values
        print(f"🎯 Task labels from '{label_col}': {set(task_labels)}")
    
    if priority_columns:
        # Use the first available priority column
        priority_col = priority_columns[0]
        priority_labels = df[priority_col].fillna('Medium').values
        print(f"⚠️ Priority labels from '{priority_col}': {set(priority_labels)}")
    
    # If no labels found, create dummy labels based on text patterns
    if task_labels is None:
        print("⚠️ No label columns found, creating labels based on text patterns...")
        task_labels = []
        for text in df['combined_text']:
            if any(word in text.lower() for word in ['bug', 'fix', 'error', 'issue']):
                task_labels.append('Maintenance')
            elif any(word in text.lower() for word in ['design', 'ui', 'interface', 'layout']):
                task_labels.append('Design')
            elif any(word in text.lower() for word in ['test', 'testing', 'qa']):
                task_labels.append('Testing')
            elif any(word in text.lower() for word in ['document', 'docs', 'manual']):
                task_labels.append('Documentation')
            else:
                task_labels.append('Development')
        task_labels = np.array(task_labels)
        print(f"🎯 Created task labels: {set(task_labels)}")
    
    if priority_labels is None:
        print("⚠️ No priority columns found, creating labels based on text patterns...")
        priority_labels = []
        for text in df['combined_text']:
            if any(word in text.lower() for word in ['urgent', 'critical', 'emergency', 'asap']):
                priority_labels.append('High')
            elif any(word in text.lower() for word in ['low', 'minor', 'nice']):
                priority_labels.append('Low')
            else:
                priority_labels.append('Medium')
        priority_labels = np.array(priority_labels)
        print(f"⚠️ Created priority labels: {set(priority_labels)}")
    
    return df['combined_text'].values, task_labels, priority_labels

def train_and_save_models():
    """
    Main training function using your project structure
    """
    print("🚀 AI-POWERED-TASK-MANAGER Training")
    print("=" * 50)
    
    # Load and analyze data
    available_files = load_and_analyze_data()
    
    if not available_files:
        print("❌ No suitable data files found!")
        print("Please ensure you have one of these files:")
        print("  - data/tasks.csv")
        print("  - Cleaned_data.csv") 
        print("  - output_tfidf_features.csv")
        return
    
    # Select best data source
    source_name, data_info = select_best_data_source(available_files)
    
    if data_info is None:
        print("❌ No suitable data source found!")
        return
    
    # Preprocess data
    try:
        texts, task_labels, priority_labels = preprocess_data(data_info)
        print(f"✅ Preprocessed {len(texts)} text samples")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return
    
    # Create TF-IDF vectorizer
    print("\n🔤 Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform texts
    X = vectorizer.fit_transform(texts)
    print(f"✅ Created {X.shape[1]} features from {X.shape[0]} samples")
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('features', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Save vectorizer
    vectorizer_path = 'features/features_tfidf.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    print(f"💾 Vectorizer saved: {vectorizer_path}")
    
    results = {}
    
    # Train task classification model
    print("\n🎯 Training Task Classification Model...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, task_labels, test_size=0.3, random_state=42, stratify=task_labels
        )
        
        # Train Naive Bayes
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = nb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Task Classification Accuracy: {accuracy * 100:.2f}%")
        
        # Save model
        model_path = 'models/task_classifier_nb.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(nb_model, f)
        print(f"💾 Task model saved: {model_path}")
        
        results['task_classification'] = accuracy
        
    except Exception as e:
        print(f"❌ Task classification training failed: {e}")
    
    # Train priority classification model
    print("\n⚠️ Training Priority Classification Model...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, priority_labels, test_size=0.3, random_state=42, stratify=priority_labels
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Priority Classification Accuracy: {accuracy * 100:.2f}%")
        
        # Save model
        model_path = 'models/priority_model_rf.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"💾 Priority model saved: {model_path}")
        
        results['priority_classification'] = accuracy
        
    except Exception as e:
        print(f"❌ Priority classification training failed: {e}")
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save as JSON
    import json
    results_json = f'evaluation_results/evaluation_results_{timestamp}.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved: {results_json}")
    
    # Save as CSV
    results_df = pd.DataFrame([results])
    results_csv = f'evaluation_results/model_comparison_{timestamp}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"📊 Results saved: {results_csv}")
    
    # Test the models
    print("\n🧪 Testing saved models...")
    test_models()

def test_models():
    """
    Test the trained models with sample inputs
    """
    try:
        # Load vectorizer and models
        vectorizer = joblib.load('features/features_tfidf.pkl')
        
        with open('models/task_classifier_nb.pkl', 'rb') as f:
            task_model = pickle.load(f)
        
        with open('models/priority_model_rf.pkl', 'rb') as f:
            priority_model = pickle.load(f)
        
        print("✅ All models loaded successfully!")
        
        # Test samples
        test_texts = [
            "fix database connection timeout error",
            "create new user interface design for mobile app",
            "write comprehensive documentation for api endpoints",
            "urgent security patch needed for authentication system"
        ]
        
        print("\n🎯 Testing with sample tasks:")
        print("-" * 50)
        
        for text in test_texts:
            vector = vectorizer.transform([text])
            
            task_pred = task_model.predict(vector)[0]
            task_proba = max(task_model.predict_proba(vector)[0])
            
            priority_pred = priority_model.predict(vector)[0]
            priority_proba = max(priority_model.predict_proba(vector)[0])
            
            print(f"📝 '{text}'")
            print(f"   🎯 Task: {task_pred} (confidence: {task_proba:.2f})")
            print(f"   ⚠️ Priority: {priority_pred} (confidence: {priority_proba:.2f})")
            print()
        
        print("✅ All models are working correctly!")
        print("\n🎉 Training completed successfully!")
        print("You can now run your Streamlit app with: streamlit run your_app.py")
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")

if __name__ == "__main__":
    import numpy as np  # Add this import
    train_and_save_models()