# i2i Academy - Applied Data Science

This repository contains an end-to-end Machine Learning classification pipeline developed as part of the i2i Academy Training Program.

## Project Overview
The objective of this project is to evaluate and compare multiple machine learning classification algorithms on a high-dimensional medical dataset (Breast Cancer Wisconsin Dataset) to accurately predict whether a tumor is malignant or benign.

## Technical Architecture & Dependencies
* **Python 3** (Fully optimized for Python 3.16)
* **Pandas**: For structural data frame representation.
* **Scikit-Learn**: For preprocessing pipelines, model training, and diagnostic metrics evaluation.

## Algorithmic Performance
The dataset was split into an 80% training set and a 20% validation set. Feature scaling was performed using `StandardScaler` to accommodate distance-based calculations.

### 1. Random Forest Classifier
* **Accuracy Score:** 96.49%
* **Confusion Matrix:**
  ```text
  [[40  3]
   [ 1 70]]

### 2. K-Nearest Neighbors (k-NN)
* **Accuracy Score:** 94.74%
* **Confusion Matrix:**
  ```text
  [[40  3]
  [ 3 68]]
  
