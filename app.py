import streamlit as st
import pandas as pd
import pickle
import joblib
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Task Manager",
    page_icon="🧠",
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
        color: white;
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
    <h1>🧠 Smart Task Manager</h1>
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
elif page == "🔍 Analyze My Task":
    
    if error_messages is not None:
        st.error("🔧 AI models need to be set up first. Please check the 'Get Started' page.")
        st.stop()
    
    st.markdown("## 🔍 Task Analysis")
    st.markdown("Describe your task below and get instant insights!")
    
    # Check if we have an example task from the home page
    default_text = ""
    if hasattr(st.session_state, 'example_task') and hasattr(st.session_state, 'go_to_analyze'):
        if st.session_state.go_to_analyze:
            default_text = st.session_state.example_task
            st.session_state.go_to_analyze = False
    
    # Task input
    task_text = st.text_area(
        "✏️ **What's your task?**",
        value=default_text,
        height=120,
        placeholder="Example: Create a presentation for the client meeting next week",
        help="Be as descriptive as you can - the more details, the better the analysis!"
    )
    
    # Analyze button
    if st.button("🚀 Analyze My Task", type="primary"):
        if not task_text.strip():
            st.warning("⚠️ Please describe your task first!")
        else:
            # Show loading
            with st.spinner("🤔 AI is thinking about your task..."):
                try:
                    # Transform and predict
                    vector = vectorizer.transform([task_text])
                    
                    # Get predictions
                    task_type = task_model.predict(vector)[0]
                    task_confidence = max(task_model.predict_proba(vector)[0])
                    
                    priority_level = priority_model.predict(vector)[0]
                    priority_confidence = max(priority_model.predict_proba(vector)[0])
                    
                    # Show results
                    st.markdown("---")
                    st.markdown("## 🎯 Analysis Results")
                    
                    # Main predictions in colorful cards
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        confidence_text = "High confidence" if task_confidence > 0.7 else "Medium confidence" if task_confidence > 0.5 else "Low confidence"
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🗂️ Task Type</h3>
                            <h2 style="color: #667eea;">{task_type}</h2>
                            <p style="margin: 0;"><small>{confidence_text}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Color code priority
                        priority_color = "#dc3545" if priority_level.lower() in ['high', 'urgent', 'critical'] else "#ffc107" if priority_level.lower() in ['medium', 'normal'] else "#28a745"
                        priority_emoji = "🔥" if priority_level.lower() in ['high', 'urgent', 'critical'] else "⚡" if priority_level.lower() in ['medium', 'normal'] else "🟢"
                        
                        confidence_text = "High confidence" if priority_confidence > 0.7 else "Medium confidence" if priority_confidence > 0.5 else "Low confidence"
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>{priority_emoji} Priority Level</h3>
                            <h2 style="color: {priority_color};">{priority_level}</h2>
                            <p style="margin: 0;"><small>{confidence_text}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Helpful tips based on results
                    st.markdown("### 💡 Smart Suggestions")
                    
                    # Task type suggestions
                    task_tips = {
                        'Development': [
                            "🔧 Break this into smaller coding tasks",
                            "📝 Document your approach before starting",
                            "🧪 Plan for testing and debugging time"
                        ],
                        'Design': [
                            "🎨 Gather inspiration and references first",
                            "👥 Get feedback early and often",
                            "📱 Consider different screen sizes and devices"
                        ],
                        'Testing': [
                            "📋 Create a test plan with clear steps",
                            "🐛 Document any bugs you find",
                            "✅ Test on different devices/browsers"
                        ],
                        'Documentation': [
                            "📚 Keep it simple and easy to understand",
                            "📸 Use screenshots and examples",
                            "🔄 Review and update regularly"
                        ],
                        'Maintenance': [
                            "⏰ Schedule regular maintenance windows",
                            "💾 Always backup before making changes",
                            "📊 Monitor the results after completion"
                        ]
                    }
                    
                    # Priority suggestions
                    priority_tips = {
                        'High': [
                            "🚨 This needs immediate attention",
                            "👥 Consider getting help or resources",
                            "📅 Block time in your calendar today"
                        ],
                        'Critical': [
                            "🔥 Drop everything else and focus on this",
                            "📞 Communicate with stakeholders immediately",
                            "🆘 Get all the help you need"
                        ],
                        'Medium': [
                            "📅 Schedule this for this week",
                            "📋 Add it to your task list",
                            "⏰ Set a reasonable deadline"
                        ],
                        'Low': [
                            "📚 This can wait for when you have free time",
                            "💡 Consider if this is really necessary",
                            "📅 Maybe schedule for next week or month"
                        ]
                    }
                    
                    # Show relevant tips
                    tips_to_show = []
                    
                    # Add task-specific tips
                    for task_key, tips in task_tips.items():
                        if task_key.lower() in task_type.lower():
                            tips_to_show.extend(tips[:2])  # Show first 2 tips
                            break
                    
                    # Add priority-specific tips
                    for priority_key, tips in priority_tips.items():
                        if priority_key.lower() in priority_level.lower():
                            tips_to_show.extend(tips[:2])  # Show first 2 tips
                            break
                    
                    # Default tips if none match
                    if not tips_to_show:
                        tips_to_show = [
                            "📋 Break complex tasks into smaller steps",
                            "⏰ Estimate how long this will take",
                            "🎯 Focus on the most important parts first"
                        ]
                    
                    # Display tips in a nice format
                    for tip in tips_to_show:
                        st.markdown(f"""
                        <div class="tips-box">
                            {tip}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Encouraging message
                    st.success("🎉 Great! Now you have a clear understanding of your task. You've got this! 💪")
                    
                except Exception as e:
                    st.error("😅 Oops! Something went wrong with the analysis. Please try again.")
                    if st.checkbox("Show technical details"):
                        st.error(f"Error: {e}")

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