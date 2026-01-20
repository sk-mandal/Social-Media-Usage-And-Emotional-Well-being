# Capstone Project – 3  
## Social Media Usage and Emotional Well-being

---

## Project Details

- **Name:** Subhash Kumar Mandal  
- **Contact No.:** 8130553392  
- **Email IDs:**  
  - subh8130@gmail.com  
  - itsmesmandal@gmail.com  

---

## 1. Introduction

In recent years, social media platforms have become an integral part of daily life, influencing not only communication patterns but also users’ emotional well-being. Understanding how social media usage behavior relates to emotional states is important for researchers, platform designers, and mental health professionals.

With the availability of structured user activity data, machine learning and deep learning techniques can be leveraged to model and predict users’ dominant emotional states based on their online behavior.

The objective of this project is to build an **end-to-end predictive system** that estimates a user’s **Dominant Emotion** based on their social media usage patterns. The project explores both **traditional machine learning (ML)** and **deep learning (DL)** approaches, compares their performance, and identifies the most suitable modeling strategy for structured tabular data.

---

## 2. Problem Statement

The primary goal of this project is to **predict the dominant emotional state** of a social media user using behavioral and demographic attributes such as:

- Age  
- Gender  
- Platform used  
- Daily usage time  
- Posts per day  
- Likes received  
- Comments received  
- Messages sent  

This is formulated as a **multi-class classification problem**, where each user belongs to exactly one emotion category (e.g., *Happiness, Anxiety, Sadness, Neutral*, etc.).

### 3. Project Objectives

- Build predictive models using both **machine learning and deep learning** techniques  
- Compare **baseline, advanced, and tuned** models  
- Analyze the impact of **class imbalance** on model performance  
- Select the **best-performing model** based on validation metrics  

---

## 4. Dataset Description

The dataset captures social media usage behavior and associated emotional states. It is split into three files to ensure **leakage-free evaluation**:

- `train.csv` – Training data  
- `val.csv` – Validation data  
- `test.csv` – Final evaluation data  

Each record represents a unique user with the following features:

| Feature | Description |
|------|------------|
| User_ID | Unique identifier (removed during modeling) |
| Age | Age of the user |
| Gender | Female, Male, Non-binary |
| Platform | Social media platform used |
| Daily_Usage_Time | Daily time spent (minutes) |
| Posts_Per_Day | Average posts per day |
| Likes_Received_Per_Day | Average likes per day |
| Comments_Received_Per_Day | Average comments per day |
| Messages_Sent_Per_Day | Messages sent per day |
| Dominant_Emotion | **Target variable** |

The target variable contains multiple emotion classes, making this a **multi-class classification task**.

---

## 5. Project Deliverables

- Cleaned and preprocessed dataset  
- Comprehensive **Exploratory Data Analysis (EDA)**  
- Baseline ML models (Dummy Classifier, Logistic Regression)  
- Advanced ML models (Decision Tree, Random Forest, Gradient Boosting)  
- **Tuned Random Forest model** (best-performing ML model)  
- Deep Learning model (Artificial Neural Network)  
- Detailed evaluation using accuracy, precision, recall, and F1-score  
- Comparative analysis between ML and DL approaches  
- Final selected model suitable for deployment  
- Well-documented and reproducible project report  

---

## 6. Software and Tools

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **TensorFlow, Keras**
- **Jupyter Notebook / Python scripts**

---

## 6. Dataset Source

**Kaggle – Social Media Usage and Emotional Well-Being Dataset**

This dataset contains structured and anonymized data suitable for academic research and machine learning experimentation.

---


## Report Findings and Analysis

### 1. Exploratory Data Analysis (EDA)

EDA was conducted to understand emotional distributions, platform-wise and gender-wise trends, numeric feature behavior, correlations, and outliers.

#### 1.1 Distribution of Dominant Emotion

- Dataset is **highly imbalanced**
- Most frequent emotions: **Happiness, Neutral**
- Rare emotions: **Anger, Aggression**
- Accuracy alone is insufficient → **class-wise metrics are critical**

#### 1.2 Emotion Distribution Across Platforms

- **Instagram:** Dominated by Happiness  
- **Twitter:** Higher Anger and Sadness  
- **Facebook:** Neutral, Anxiety, Boredom  
- **LinkedIn:** Strongly associated with Boredom and Neutral  
- **WhatsApp / Telegram:** Balanced with slightly higher Neutral and Anxiety  

This confirms that **platform type significantly influences emotional state**.

#### 1.3 Emotion Distribution by Gender

- Male and Female users show higher Happiness, Neutral, and Anxiety  
- Non-binary users show a more even distribution (smaller sample size)  
- Gender is a meaningful categorical feature

#### 1.4 Numerical Feature Distribution

- Right-skewed distributions for:
  - Daily usage time  
  - Likes, comments, messages  
- Indicates varying engagement intensity

#### 1.5 Outlier Analysis

- Outliers represent **genuine heavy-user behavior**
- Outliers were retained
- Scaling techniques used instead of removal

#### 1.6 Correlation Analysis

- Moderate correlation among engagement metrics
- No severe multicollinearity
- All numerical features retained

---

## Machine Learning Models

### 2.1 Baseline Models

| Model | Validation Accuracy |
|----|----|
| Dummy Classifier | 18.75% |
| Logistic Regression | ~51.4% |

Baseline results highlight the **complexity of the task** and the impact of imbalance.

### 2.2 Advanced ML Models

| Model | Validation Accuracy |
|----|----|
| Decision Tree | 81.25% |
| Random Forest | 81.94% |
| Gradient Boosting | 80.56% |

- Decision Tree showed overfitting
- Random Forest provided better generalization

### 2.3 Hyperparameter Tuning

- **Tuned Random Forest** achieved **82.64% accuracy**
- Best balance between bias and variance
- Tuned Gradient Boosting did not improve further

---

## Deep Learning Model

A feedforward **Artificial Neural Network (ANN)** was implemented:

- Class-weighted training used
- Validation accuracy: **~65.3%**
- Improved minority recall slightly
- Still underperformed compared to ensemble ML models

**Key Insight:**  
Deep learning is **not always superior**, especially for structured tabular data with limited size.

---

## Model Comparison and Final Selection

- Ensemble ML models outperformed both baseline and DL models
- **Tuned Random Forest** achieved the best overall performance
- Selected as the **final model** for deployment

---

## Conclusion

This project demonstrates a complete end-to-end workflow for predicting emotional well-being using social media usage data. Exploratory analysis revealed strong behavioral and platform-based patterns. Model comparison showed that **ensemble-based machine learning models** are more effective than deep learning for this dataset.

The **Tuned Random Forest model** emerged as the most accurate and robust solution, making it the ideal choice for future deployment or further research.

---
