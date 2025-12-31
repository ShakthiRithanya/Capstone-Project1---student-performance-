import gradio as gr
import joblib
import pandas as pd
import os

# Paths
MODEL_PATH = "student_performance_model.pkl"
COLUMNS_PATH = "feature_columns.pkl"

# Custom CSS for a Premium Professional Look
custom_css = """
footer {visibility: hidden}
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
#main-title {
    text-align: center;
    color: #1e293b;
    font-weight: 800;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem;
}
#sub-title {
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.sidebar-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.predict-btn {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 15px !important;
    transition: all 0.3s ease !important;
}
.predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
}
.output-box {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    background: white !important;
}
"""

def predict_performance(studytime, failures, absences, goout, freetime, health, G1, G2):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        return "⚠️ Setup Incomplete: Please run the training notebook first."
    
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(COLUMNS_PATH)
        
        input_data = pd.DataFrame([[
            studytime, failures, absences, goout, freetime, health, G1, G2
        ]], columns=feature_columns)
        
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        status = "🎓 PASS" if prediction == 1 else "❌ FAIL"
        confidence = f"Confidence: {prob*100:.1f}%" if prediction == 1 else f"Confidence: {(1-prob)*100:.1f}%"
        
        return f"{status}\n{confidence}"
    except Exception as e:
        return f"Error: {str(e)}"

# Custom Theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    section_header_text_color="*primary_600",
    block_title_text_weight="600",
    block_label_text_weight="600",
)

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("# 🎓 EduVance: Student Metrics Intelligence", elem_id="main-title")
    gr.Markdown("Elevating educational outcomes through predictive analytics and feature-driven insights.", elem_id="sub-title")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Academic History")
            g1 = gr.Slider(0, 20, value=10, step=1, label="First Term Grade (G1)")
            g2 = gr.Slider(0, 20, value=10, step=1, label="Second Term Grade (G2)")
            failures = gr.Number(value=0, label="Previous Class Failures", precision=0)
            
        with gr.Column(scale=1):
            gr.Markdown("### 🕒 Behavioral Inputs")
            study = gr.Slider(1, 4, value=2, step=1, label="Weekly Study Time (Scale 1-4)")
            absences = gr.Slider(0, 93, value=0, step=1, label="Number of Absences")
            goout = gr.Slider(1, 5, value=2, step=1, label="Going Out Frequency")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🩺 Personal Metrics")
            free = gr.Slider(1, 5, value=3, step=1, label="Free Time Availability")
            health = gr.Slider(1, 5, value=3, step=1, label="Current Health Status")
            
    with gr.Row():
        predict_btn = gr.Button("🚀 GENERATE PREDICTION", variant="primary", elem_classes="predict-btn")
        
    with gr.Row():
        output = gr.Textbox(label="AI Analysis Result", elem_classes="output-box", lines=2)

    predict_btn.click(
        fn=predict_performance,
        inputs=[study, failures, absences, goout, free, health, g1, g2],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ About EduVance
    This system utilizes a **Random Forest Ensemble** model trained on the UCI Student Performance dataset. 
    It achieves an average accuracy of **88.6%** in predicting final outcomes based on multi-dimensional feature inputs.
    """)

if __name__ == "__main__":
    demo.launch()
