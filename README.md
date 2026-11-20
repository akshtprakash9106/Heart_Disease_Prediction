
# ❤️ Heart Disease Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts the likelihood of heart disease based on patient health parameters using a Random Forest Classifier.

The project covers data preprocessing, exploratory data analysis, model comparison, hyperparameter optimization, probability threshold tuning, and deployment through a Streamlit web application.

## 🚀 Live Application

Coming soon!

## 📌 Project Overview

Heart disease is a major global health concern. This project explores how machine learning classification algorithms can identify patterns in patient health data and estimate the likelihood of heart disease.

A binary classification approach is used:
- `0` – No heart disease
- `1` – Heart disease present

**Note:** This is an educational project and is not intended for clinical diagnosis or medical decision-making.

## 📊 Dataset

- **Source:** [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Records:** 920
- **Original features:** 16
- **Target variable:** `num`

The original target was converted into a binary classification target:
- `num = 0` → No heart disease
- `num > 0` → Heart disease present

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## ⚙️ Project Workflow

1. Data loading and inspection
2. Data cleaning and missing value treatment
3. Exploratory Data Analysis (EDA)
4. Feature engineering and target conversion
5. Train-test split with stratification
6. Numerical feature scaling
7. Categorical feature encoding
8. Model training and comparison
9. Random Forest hyperparameter optimization
10. Probability threshold optimization
11. Model evaluation
12. Streamlit application development

## 🤖 Models Evaluated

The following classification algorithms were compared:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Random Forest was selected for further optimization based on its cross-validation performance.

## 📈 Final Model Performance

The optimized Random Forest model was evaluated on a held-out test set using a classification threshold of approximately `0.32`.

| Metric | Score |
|---|---:|
| Accuracy | 82.07% |
| Precision | 75.94% |
| Recall | 99.02% |
| F2 Score | 93.35% |

The threshold was optimized using out-of-fold predictions on the training data, with F2-score as the selection metric.

The model prioritizes recall, reducing the number of positive heart disease cases missed, at the cost of more false positives.

These results are from a single train-test split and have not been independently clinically validated.

## 💻 Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/akshtprakash9106/Heart_Disease_Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Heart_Disease_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Repository Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── heart_disease_model.pkl
├── model_threshold.json
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Feature importance visualization
- Model performance dashboard
- Improved model explainability
- Independent external validation
- Cloud deployment

## 👨‍💻 Author

Developed as a Machine Learning project using Python and Streamlit.
