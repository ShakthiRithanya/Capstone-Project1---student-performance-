Predicting Student Performance Using Decision Trees & Random Forests
Overview

This project uses machine learning techniques to predict whether a student will Pass or Fail based on academic, behavioral, and demographic data.
The system leverages Decision Tree and Random Forest classifiers to identify key factors influencing academic success and provides predictions through a Gradio-based web interface.

Project Objectives

Build a binary classification model (Pass/Fail).

Apply Decision Tree and Random Forest algorithms.

Identify key features influencing student performance.

Deploy the final model using an interactive Gradio UI.

Dataset Description

Source: Student Performance Dataset from the UCI Machine Learning Repository (via Kaggle).

Attributes: Includes grades, study time, absences, failures, and lifestyle factors.

Target Variable:

Pass → G3 ≥ 10

Fail → G3 < 10

This converts the problem into a binary classification task.

Exploratory Data Analysis (EDA)

Key observations:

Previous grades (G1, G2) are highly correlated with final performance.

More study time improves success rates.

High absences and prior failures negatively impact results.

EDA was used to identify the most relevant features for model training.

Data Preprocessing

Steps included:

Converting G3 into a binary target (pass_fail).

Encoding categorical variables.

Selecting key input features:
studytime, failures, absences, goout, freetime, health, G1, G2

Splitting data (80% training / 20% testing) with stratified sampling.

Model Development
1. Decision Tree Classifier

Interpretable model with clear decision paths.

Susceptible to overfitting on complex datasets.

2. Random Forest Classifier

Ensemble of decision trees that enhances generalization.

Reduces overfitting and provides better accuracy.

Supports feature importance analysis.

The Random Forest model was chosen for final deployment.

Model Evaluation

Metrics Used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Results:

Random Forest outperformed the Decision Tree.

Balanced precision and recall.

Fewer false predictions in confusion matrix.

Feature Importance

Top features influencing student success:

Previous grades (G1, G2)

Number of past failures

Study time

Absences

User Interface (Gradio)

A Gradio web app was created for user interaction.

Features:

Sliders for academic & behavioral inputs.

Real-time prediction output (Pass/Fail).

Simple, user-friendly interface.

Consistent with trained model’s input features.

Conclusion

This project demonstrates a full machine learning workflow—from data exploration and preprocessing to model training, evaluation, and deployment.
The Random Forest model achieved high accuracy and interpretability, making it ideal for real-world educational analytics.

How to Run
1. Clone Repository
git clone https://github.com/ShakthiRithanya/Capstone-Project1---student-performance-.git
cd Capstone-Project1---student-performance-

2. Install Dependencies
pip install -r requirements.txt

3. Run the Application
python app.py

4. Access the Gradio Interface

After running the script, open the local URL (shown in terminal) to interact with the model.

Requirements
pandas
numpy
scikit-learn
gradio
matplotlib
seaborn
