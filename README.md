# 🏥 Predicting Hospital Readmission Rates

This project demonstrates how to predict hospital readmission within 30 days using **synthetic data** and multiple machine learning models. The pipeline includes classical ML algorithms and a deep learning approach using Keras (TensorFlow).

---

## 📦 Project Structure

- `hospital_readmission_prediction.ipynb` – Jupyter Notebook with full end-to-end workflow.
- `predict_hospital_readmission.py` – Python script version of the same workflow.
- `roc_curve_comparison.png` – Visualization of ROC curves comparing model performance.
- `hospital_readmission_report.txt` – Summary report of results and findings.

---

## 🧠 Models Included

| Model                 | Description                             |
|----------------------|-----------------------------------------|
| Logistic Regression  | Baseline linear model                   |
| Random Forest        | Ensemble-based model using decision trees |
| XGBoost              | Gradient boosting algorithm             |
| MLP (Keras)          | Deep learning model with dense layers   |

---

## 🗂 Features Used (Synthetic)

- Age  
- Gender  
- Length of Stay  
- Number of Previous Admissions  
- Comorbidity Score  
- Diabetes Status  
- Hypertension Status  
- Discharge Destination (Home or Not)

---

## ⚙️ How to Run

1. Clone this repo or download the `.ipynb` and `.py` files.
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn

    Run the notebook:

jupyter notebook hospital_readmission_prediction.ipynb

Or execute the script:

python predict_hospital_readmission.py

📊 Output

    Evaluation metrics: Accuracy, ROC AUC, Precision, Recall, F1-score

    Visualization: Combined ROC Curve Comparison

    Summary report with key findings and future recommendations.

📌 Summary of Findings

    XGBoost and MLP performed best in terms of ROC AUC.

    MLP handled non-linear relationships effectively.

    Logistic Regression provided fast and interpretable results.

    The workflow is extendable to real-world EHR datasets like MIMIC-III.

📈 Future Work

    Incorporate real hospital datasets.

    Apply feature selection and hyperparameter tuning.

    Deploy as an API or interactive app using Flask or Streamlit.

👨‍⚕️ Author


Data Scientist & Health Tech Enthusiast
📧 Contact me | 🌐 LinkedIn
