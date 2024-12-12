# Pump It Up: Data Mining the Water Table - DrivenData Competition

This repository contains my submission for the DrivenData "Pump It Up: Data Mining the Water Table" competition. The challenge involves predicting the operational status of water pumps in Tanzania using machine learning. By leveraging historical data, this project builds a model to classify pumps as functional, non-functional, or needing repair, providing actionable insights for better water management and infrastructure planning. The final results of running this ML Ensemble Model (using Random Forest, XGBoost, and CatBoost) will be an Excel sheet that can be submitted into the competition. To download the data required for this model, please see the [DrivenData Competition Site](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/).

## 📑 **Table of Contents**

1. [Business Value](#-business-value)
2. [Technologies Used](#-technologies-used)
3. [What the Code Does](#-what-the-code-does)
4. [Key Highlights](#-key-highlights)

---

## 🌍 **Business Value**

Access to clean water is a fundamental necessity, and the operational status of water pumps directly impacts the livelihoods of millions of people. By accurately predicting the status of water pumps, this project aims to:

- **Improve resource allocation**: Help NGOs and local governments identify areas where repairs or maintenance are needed, ensuring a more efficient use of resources.
- **Enhance water access**: By identifying non-functional pumps early, it enables timely repairs, reducing water scarcity and improving health outcomes.
- **Scale impact**: A reliable prediction model can be deployed across various regions, assisting in the global effort to provide clean water to underserved communities.

This project showcases the application of machine learning to real-world problems that have a significant social and economic impact, contributing to the achievement of sustainable development goals.

---

## 🛠️ **Technologies Used**

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **XGBoost, CatBoost, RandomForest**
- **Optuna, Grid Search, Random Search**
- **Matplotlib & Seaborn** (for visualization)
- **Jupyter Notebook** (for model development)
- **Git & GitHub** (for version control)

---

## 🔎 **What the Code Does**

There are two versions of the code:

- To run the classification system, use the [src/ClassificationSystem.ipynb](src/ClassificationSystem.ipynb) file.
- If you want to use the [src/ClassificationSystem_WithOutputs.ipynb](src/ClassificationSystem_WithOutputs.ipynb) file, you may need to download it first (instead of viewing directly on GitHub) due to the file size.

The project is organized into the following key components:

1. **Data Preprocessing**:  
   - Cleaning and imputing missing values to ensure a complete and accurate dataset.  
   - Encoding categorical variables to make them suitable for machine learning models.  
   - Scaling numerical features for consistency and improved model performance.  

2. **Exploratory Data Analysis (EDA)**:  
   - Analyzing distributions, relationships, and trends within the data to gain critical insights.  
   - Visualizing key patterns and anomalies to inform subsequent feature engineering decisions.  

3. **Feature Engineering**:  
   - Creating new, domain-relevant features to capture additional predictive signals.  
   - Refining and selecting the most impactful features to enhance model performance and generalization.  

4. **Model Training**:  
   - Utilizing diverse algorithms, including Random Forest, XGBoost, and CatBoost, to capture different aspects of the data.  
   - Applying advanced hyperparameter tuning techniques such as Optuna, GridSearch, and RandomSearch to optimize performance.  

5. **Evaluation**:  
   - Using comprehensive metrics like accuracy, precision, recall, F1-score, and confusion matrices to evaluate model effectiveness.  
   - Comparing performance across models to select the best solution for submission.

---

## 🚀 **Key Highlights**

- Achieved **82.19% accuracy** on the test set, demonstrating strong performance in classifying pump functionality.
- Developed a pipeline for preprocessing, training, and evaluating models for future scalability and deployment.
- Visualizations and insights from EDA guide impactful feature creation and decision-making.