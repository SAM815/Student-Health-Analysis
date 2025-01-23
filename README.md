# Student Health Analysis - Machine Learning Project

## Overview
This project aims to predict student health risks and self-reported stress levels using the Student Health Data dataset from Kaggle. The dataset includes various physiological, psychological, and behavioral metrics for college students, and the goal is to use machine learning to evaluate and predict health risks and stress levels in high-stress academic environments.

The project is split into two main objectives:
1. **Predict Health Risk Levels:** A supervised learning approach to classify students into health risk categories (Low, Moderate, High).
2. **Predict Self-Reported Stress Levels:** A supervised regression problem aimed at predicting stress levels based on various health and lifestyle indicators.

## Dataset Overview
The dataset, sourced from Kaggle, contains information on:
- **Demographic Data**: Age, gender, student ID
- **Physiological Data**: Heart rate, blood pressure (systolic/diastolic), stress levels from biosensor readings
- **Psychological Data**: Self-reported stress levels, mood states
- **Academic and Behavioral Data**: Study hours, physical activity, sleep quality
- **Target Variables**:
  - **Health Risk Level**: (Low, Moderate, High) - Supervised classification
  - **Self-Reported Stress Level**: Continuous values representing stress (Regression problem)

## Problem Statement
We aim to address the following key questions through machine learning models:
- Can we predict a student’s health risk category based on health-related features?
- What factors (features) are most important in predicting health risk and stress levels?
- What machine learning algorithms (KNN, Random Forest, Decision Trees, etc.) provide the best performance for these predictions?
- How do lifestyle factors, such as sleep quality and physical activity, affect self-reported stress levels?
- What are the key physiological indicators associated with stress?

## Approach
- **Health Risk Level Prediction**: Supervised classification model to predict health risk based on student data. We will train and evaluate classifiers like Logistic Regression, KNN, Decision Trees, and Random Forest.
- **Self-Reported Stress Prediction**: Supervised regression model to predict stress levels using linear and ensemble regression models.


## Task Breakdown
- **Target Variable**: Health Risk Level (Categorical)
- **Tasks**:
  - Conducted Exploratory Data Analysis (EDA) to understand dataset characteristics.
  - Preprocessed the data (missing value imputation, encoding categorical variables, and scaling).
  - Implemented classification models: Logistic Regression, KNN, Decision Tree, Random Forest, and others.
  - Evaluated models using metrics like accuracy, precision, recall, F1 score, and confusion matrix.
  - Performed feature importance analysis and generated model comparison plots.

- **Target Variable**: Self-Reported Stress Level (Continuous)
- **Tasks**:
  - Conducted EDA to uncover relationships and patterns in the dataset.
  - Preprocessed the data (handling missing values, encoding categorical variables).
  - Implemented regression models: Linear Regression, Random Forest Regressor, Decision Tree Regressor, and others.
  - Evaluated models using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 Score.
  - Generated prediction vs. actual plots, residual plots, and error distribution plots.

## Techniques Used
- **Data Preprocessing**: Handling missing values, feature scaling, encoding categorical variables.
- **Exploratory Data Analysis**: Data visualization using histograms, scatterplots, heatmaps, and boxplots to uncover patterns and relationships.
- **Modeling**:
  - **Classification** (Health Risk Level): Logistic Regression, KNN, Decision Trees, Random Forest, SVM, and ensemble models.
  - **Regression** (Self-Reported Stress): Linear Regression, Random Forest Regressor, Decision Tree Regressor, Support Vector Regressor.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score (for classification), MSE, MAE, R2 Score (for regression).
- **Feature Importance**: Ranking features using tree-based models and statistical analysis.

## Results & Insights
- **Health Risk Prediction**: Key features impacting health risk levels include stress levels (both objective and self-reported), sleep quality, and physical activity.
- **Stress Level Prediction**: Significant factors influencing self-reported stress include sleep quality, physical activity, and study hours.
- Ensemble models (e.g., Random Forest and Voting Classifiers) provided the best performance in both classification and regression tasks.

## Conclusion
This project successfully demonstrates how machine learning can be applied to predict health risks and self-reported stress levels in college students. The findings can be used to identify at-risk students and provide valuable insights for improving student well-being and academic performance.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

