# SC1015 Data Science & AI Mini Project - Diabetes Health Indicators

## Project Overview

This repository contains the machine learning project developed by students from Nanyang Technological University for the SC1015 - Introduction to Data Science & Artificial Intelligence course. The project aims to predict the risk of diabetes using demographic, lifestyle, and health indicator data. The dataset is sourced from the 2015 Behavioral Risk Factor Surveillance System (BRFSS) and is available on Kaggle.

## Team Members

- Dong Zhijian
- Muhammad Faqih Akmal
- Lim Zhen Rong

## Motivation

With an expected rise in diabetes cases to 1.31 billion by 2050, early and effective diabetes detection is crucial. This project develops a machine learning model to identify individuals at high risk of developing diabetes, facilitating early interventions.

## Data Source

The data used can be found here: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## Files and Folders

- `diabetes_012_health_indicators_BRFSS2015.csv`: Dataset file.
- `SC1015_Project.ipynb`: Jupyter notebook containing all the code and analysis.

## Distribution Summary

We note that the distribution of the target variable, which categorizes respondents into 'No diabetes', 'Prediabetes', and 'Diabetes', has a noticeable data imbalance, with a significantly larger number of respondents identified as non-diabetic compared to those categorized as pre-diabetic or diabetic. 

Moreover, other variables that could be predictors for diabetes, such as BMI, also show signs of data imbalance and outliers. The BMI distribution, for instance, has a long tail with high values that could represent extreme cases of obesity, a known risk factor for diabetes. If these outliers represent true data points, they could be crucial in predicting diabetes, but if they are errors, they may distort the model's accuracy.

The categorical variables related to lifestyle and health behaviors such as physical activity, fruit and vegetable consumption, and heavy alcohol consumption could also influence the prediction model. However, the presence of outliers in the behavioral categories, especially in heavy alcohol consumption, might require careful handling to avoid skewing the results.

## Handling the Problem of Data Imbalance using SMOTE & RandomUnderSampler

We used resampling techniques to combat class imbalance in the dataset. We divide the features and target variable into training and testing sets. 75% of the data is used for training the model, and 25% is reserved for testing.

The code combines Synthetic Minority Over-sampling Technique (SMOTE) and Random Undersampling into a pipeline to address class imbalance. SMOTE is used to oversample the minority classes, while Random Undersampling reduces the size of the majority class. This strategy aims to improve model performance by ensuring a more balanced class distribution.

## Summary

The first model tested is logistic regression, fine-tuned to run a high number of times to ensure it works as best as it can. Once the model is fitted, it's tested on a single piece of data from the test set, and the model's certainty about that data belonging to each possible category is shown.

Accuracy:

XGBoost...

Accuracy:

XVM...

Accuracy:

Random Forest...

Accuracy:



