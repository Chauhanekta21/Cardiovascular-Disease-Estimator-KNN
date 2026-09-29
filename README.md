# Cardiovascular Disease Prediction - KNN

# Cardiovascular Disease Prediction using KNN

A machine learning project using **K-Nearest Neighbors (KNN)** to predict the presence of cardiovascular disease based on patient health and lifestyle data.

## Dataset

* **Rows:** 70,000
* **Columns:** 13
* **Target:** Cardiovascular disease (`0 = No CVD`, `1 = CVD`)
* **Features:** Age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol intake, and physical activity.

## Project Workflow

### 1. Data Inspection

* Checked dataset shape, data types, missing values, and duplicates.
* No missing values were found.
* No exact duplicate rows were found.
* Examined distributions and extreme values in numerical features.
* Checked categorical and binary feature values.

### 2. Data Cleaning

Identified clearly invalid or suspicious measurements and applied practical thresholds:

| Feature      | Cleaning Rule           |
| ------------ | ----------------------- |
| Height       | Keep 100–220 cm         |
| Weight       | Keep ≥ 30 kg            |
| Systolic BP  | Keep > 0 and ≤ 250 mmHg |
| Diastolic BP | Keep > 0 and < 250 mmHg |

Additionally, rows where **systolic BP ≤ diastolic BP** were removed.

Age was converted from days to whole years using 365.25 days per year.

After cleaning:

* **Original rows:** 70,000
* **Final rows:** 68,675
* **Rows removed:** 1,325

### 3. Feature Selection

Used multiple statistical approaches rather than relying only on correlation.

**Correlation analysis**

* Systolic BP showed the strongest linear correlation with CVD.
* Age, diastolic BP, cholesterol, and weight also showed positive correlations.

**Mann–Whitney U test**

* Numerical features: age, height, weight, systolic BP, and diastolic BP.
* All showed statistically significant differences between the CVD and No-CVD groups.

**Chi-square test**

* Tested categorical/binary features against the target.

**Cramér's V**

* Cholesterol showed the strongest categorical association with CVD.
* Gender showed a very weak association and was removed.

The `id` column was also removed because it is only an identifier.

### 4. Train-Test Split

Used an **80/20 split** with stratification:

* Training set: 80%
* Test set: 20%
* `random_state = 42`
* `stratify = y`

Stratification preserves the target-class distribution in both datasets.

### 5. Feature Scaling

Applied **StandardScaler** because KNN is distance-based.

The scaler was fitted only on the training data and then applied to the test data to avoid data leakage.

### 6. KNN Model

Tested different values of **K** using 5-fold cross-validation on the training data.

The initial preprocessing produced:

* **Best K:** 99
* **CV Accuracy:** 73.08%
* **Test Accuracy:** 72.45%

The model achieved approximately:

* CVD Precision: 75%
* CVD Recall: 66%
* CVD F1-score: 70%

### 7. Preprocessing Experiment

Tested one-hot encoding for the ordinal categorical features:

* Cholesterol level
* Glucose level

Binary features remained as 0/1.

After re-selecting K using cross-validation:

* **Best K:** 121
* **CV Accuracy:** 73.34%
* **Test Accuracy:** 72.60%
* **CVD Precision:** 75%
* **CVD Recall:** 66%
* **CVD F1-score:** 71%

This preprocessing change produced only a small improvement, so further preprocessing experiments are still possible.

## Current Result

The current KNN model achieves approximately **72.6% test accuracy** with a **66% recall for CVD**.

The project is still in the model improvement stage, with further experiments planned for preprocessing and KNN optimization.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
Cardiovascular_Disease_Prediction_KNN/
│
├── data/
│   └── cardio_train.csv
│
├── jupyter_notebook/
│   └── cardiovascular_disease_knn.ipynb
│
└── README.md
```
