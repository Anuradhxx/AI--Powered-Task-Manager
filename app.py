import streamlit as st
import pandas as pd
import pickle
import joblib
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Task Manager",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .tips-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: pink;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models Function ---
@st.cache_data
def load_ai_models():
    """Load the AI models quietly without showing technical details"""
    
    try:
        # Check if model files exist
        vectorizer_path = 'features/features_tfidf.pkl'
        task_model_path = 'models/task_classifier_nb.pkl'
        priority_model_path = 'models/priority_model_rf.pkl'
        
        missing_files = []
        if not os.path.exists(vectorizer_path):
            missing_files.append("Text Analyzer")
        if not os.path.exists(task_model_path):
            missing_files.append("Task Classifier")
        if not os.path.exists(priority_model_path):
            missing_files.append("Priority Predictor")
        
        if missing_files:
            return None, None, None, missing_files
        
        # Load models
        vectorizer = joblib.load(vectorizer_path)
        
        with open(task_model_path, 'rb') as f:
            task_model = pickle.load(f)
        
        with open(priority_model_path, 'rb') as f:
            priority_model = pickle.load(f)
        
        # Quick compatibility test
        test_vector = vectorizer.transform(["test task"])
        task_model.predict(test_vector)
        priority_model.predict(test_vector)
        
        return task_model, priority_model, vectorizer, None
    
    except Exception as e:
        return None, None, None, [f"Loading error: {str(e)[:50]}..."]

# --- Load Models ---
task_model, priority_model, vectorizer, error_messages = load_ai_models()

# --- Main App ---
st.markdown("""
<div class="main-header">
    <h1>📝 Smart Task Manager</h1>
    <p>Let AI help you organize and prioritize your tasks!</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("### 🚀 Navigation")
    page = st.radio(
        "Choose what you'd like to do:",
        ["🏠 Get Started", "🔍 Analyze My Task", "📊 How Well Does It Work?"],
        label_visibility="collapsed"
    )
    
    # Status indicator
    if error_messages is None:
        st.success("✅ AI is ready!")
    else:
        st.error("❌ AI needs setup")
        with st.expander("What's missing?"):
            for error in error_messages:
                st.write(f"• {error}")

# --- HOME PAGE ---
if page == "🏠 Get Started":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome! 👋")
        st.markdown("""
        **This smart tool helps you:**
        
        🎯 **Understand your tasks better** - Find out what type of work it is
        
        ⚡ **Set the right priority** - Know if it's urgent or can wait
        
        🚀 **Stay organized** - Make better decisions about your time
        """)
        
        st.markdown("### 📝 How it works:")
        st.markdown("""
        1. **Write your task** in simple words
        2. **Get instant analysis** about what type of task it is
        3. **See the priority level** so you know when to do it
        4. **Get helpful tips** on how to handle it
        """)
    
    
    

# --- TASK ANALYSIS PAGE ---
# --- TASK ANALYSIS PAGE ---
elif page == "🔍 Analyze My Task":

    if error_messages is not None:
        st.error("🔧 AI models need to be set up first. Please check the 'Get Started' page.")
        st.stop()

    st.markdown("## 🔍 Multi-Task Analyzer")
    st.markdown("Enter multiple tasks (one per line). AI will classify and prioritize each. You can also set your own priority if needed.")

    task_text_area = st.text_area(
        "📝 Enter your tasks (one per line):",
        height=200,
        placeholder="Example:\nCreate a client report\nFix login bug\nUpdate documentation"
    )

    analyze_button = st.button("🚀 Analyze Tasks")

    if analyze_button:
        tasks = [t.strip() for t in task_text_area.strip().split("\n") if t.strip()]
        if not tasks:
            st.warning("⚠️ Please enter at least one task.")
        else:
            results = []  # store each task result
            
            for i, task_text in enumerate(tasks, 1):
                st.markdown(f"### 🧾 Task {i}")
                with st.spinner(f"Analyzing Task {i}: {task_text[:40]}..."):
                    try:
                        vector = vectorizer.transform([task_text])
                        task_type = task_model.predict(vector)[0]
                        task_conf = max(task_model.predict_proba(vector)[0])

                        priority_pred = priority_model.predict(vector)[0]
                        priority_conf = max(priority_model.predict_proba(vector)[0])

                        col1, col2 = st.columns(2)

                        with col1:
                            conf_label = "High" if task_conf > 0.7 else "Medium" if task_conf > 0.5 else "Low"
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h4>🗂️ Predicted Task Type</h4>
                                <h3 style="color: #667eea;">{task_type}</h3>
                                <p><small>{conf_label} confidence</small></p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            prio_label = "High" if priority_conf > 0.7 else "Medium" if priority_conf > 0.5 else "Low"
                            color_map = {
                                'high': "#dc3545", 'critical': "#dc3545",
                                'medium': "#ffc107", 'normal': "#ffc107",
                                'low': "#28a745"
                            }
                            prio_color = color_map.get(priority_pred.lower(), "#6c757d")
                            emoji = "🔥" if "high" in priority_pred.lower() or "critical" in priority_pred.lower() else "⚡" if "medium" in priority_pred.lower() else "🟢"

                            st.markdown(f"""
                            <div class="prediction-box">
                                <h4>{emoji} Predicted Priority</h4>
                                <h3 style="color: {prio_color};">{priority_pred}</h3>
                                <p><small>{prio_label} confidence</small></p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Manual priority override
                        st.markdown("#### ✍️ Set Your Own Priority")
                        custom_priority = st.selectbox(
                            f"Override priority for Task {i}?",
                            ["Use AI Prediction", "High", "Medium", "Low"],
                            key=f"custom_prio_{i}"
                        )

                        final_priority = priority_pred if custom_priority == "Use AI Prediction" else custom_priority
                        st.markdown(f"✅ **Final Priority**: `{final_priority}`")
                        st.markdown("---")

                        # Append final result
                        results.append({
                            "Task": task_text,
                            "Predicted Type": task_type,
                            "Predicted Priority": priority_pred,
                            "Final Priority": final_priority
                        })

                    except Exception as e:
                        st.error(f"Error analyzing task {i}: {str(e)}")

            # Show summary table
            if results:
                st.markdown("## 📋 Summary of All Tasks")
                df = pd.DataFrame(results)
                st.dataframe(df)

                # Optional: allow download as CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="task_analysis_results.csv",
                    mime="text/csv"
                )


# --- PERFORMANCE PAGE ---
elif page == "📊 How Well Does It Work?":
    
    st.markdown("## 📊 Performance Overview")
    st.markdown("See how accurate and reliable our AI predictions are!")
    
    # Try to load evaluation results
    eval_data = None
    eval_files = []
    
    if os.path.exists('evaluation_results'):
        for file in os.listdir('evaluation_results'):
            if file.endswith('.csv'):
                eval_files.append(os.path.join('evaluation_results', file))
    
    if eval_files:
        try:
            # Load the most recent evaluation file
            latest_file = max(eval_files, key=os.path.getmtime)
            eval_data = pd.read_csv(latest_file)
            
            st.success("✅ Found performance data!")
            
            # Show accuracy in a user-friendly way
            st.markdown("### 🎯 Accuracy Scores")
            st.markdown("*How often our AI gets it right:*")
            
            # Create visual metrics
            col1, col2 = st.columns(2)
            
            if 'task_classification' in eval_data.columns:
                task_accuracy = eval_data['task_classification'].iloc[0]
                with col1:
                    st.metric(
                        "🗂️ Task Type Prediction",
                        f"{task_accuracy*100:.1f}%",
                        help="How accurately we identify what type of task it is"
                    )
                    
                    # Visual indicator
                    if task_accuracy > 0.8:
                        st.success("Excellent accuracy! 🌟")
                    elif task_accuracy > 0.6:
                        st.warning("Good accuracy 👍")
                    else:
                        st.info("Learning and improving 📈")
            
            if 'priority_classification' in eval_data.columns:
                priority_accuracy = eval_data['priority_classification'].iloc[0]
                with col2:
                    st.metric(
                        "⚡ Priority Level Prediction", 
                        f"{priority_accuracy*100:.1f}%",
                        help="How accurately we determine task priority"
                    )
                    
                    # Visual indicator
                    if priority_accuracy > 0.8:
                        st.success("Excellent accuracy! 🌟")
                    elif priority_accuracy > 0.6:
                        st.warning("Good accuracy 👍")
                    else:
                        st.info("Learning and improving 📈")
            
            # Overall performance summary
            st.markdown("---")
            st.markdown("### 📈 What This Means")
            
            avg_accuracy = eval_data.mean().mean() if len(eval_data.columns) > 0 else 0
            
            if avg_accuracy > 0.8:
                st.success("""
                🌟 **Excellent Performance!**
                
                Our AI is highly reliable and gives very accurate predictions. You can trust the results with confidence!
                """)
            elif avg_accuracy > 0.6:
                st.warning("""
                👍 **Good Performance!**
                
                Our AI gives good predictions most of the time. The results are quite reliable for making decisions.
                """)
            else:
                st.info("""
                📈 **Improving Performance!**
                
                Our AI is still learning and getting better. Use the predictions as a helpful guide, but also trust your judgment.
                """)
            
            # Simple chart
            if len(eval_data.columns) > 0:
                st.markdown("### 📊 Visual Summary")
                chart_data = eval_data.iloc[0] * 100  # Convert to percentages
                st.bar_chart(chart_data)
        
        except Exception as e:
            eval_data = None
    
    if eval_data is None:
        st.info("""
        📊 **No performance data available yet.**
        
        The AI models need to be trained first to generate performance statistics.
        
        Run the training script to see how well the AI performs:
        """)
        st.code("python code.py", language="bash")
    
    # FAQ Section
    st.markdown("---")
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("How accurate is the AI?"):
        st.write("""
        The AI accuracy depends on the training data quality. Generally:
        - **Above 80%**: Excellent - Very reliable predictions
        - **60-80%**: Good - Reliable for most decisions  
        - **Below 60%**: Learning - Use as guidance along with your judgment
        """)
    
    with st.expander("What if the AI gets it wrong?"):
        st.write("""
        No AI is 100% perfect! Here's what to do:
        - **Trust your experience** - You know your work best
        - **Use AI as a helpful guide** - Not the final decision
        - **The more you use it, the better it gets** - AI learns from patterns
        """)
    
    

# --- Footer ---
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)
