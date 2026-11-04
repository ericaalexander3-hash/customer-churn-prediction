# customer-churn-prediction
IN503
# Customer Churn Prediction

## Project Description

Customer churn is costly because retaining a customer is generally cheaper than acquiring a new one. This project walks through a complete data science workflow:

1. Data exploration and quality checks (including hidden blank values in `TotalCharges`)
2. Five visualizations of churn drivers
3. A reproducible scikit-learn preprocessing pipeline (scaling + one-hot encoding)
4. Logistic regression training with a stratified 80/20 train/test split
5. Evaluation (accuracy, confusion matrix, precision/recall) and coefficient interpretation
6. Model deployment simulation: save the pipeline with `joblib`, reload it, and predict on a new customer

## Installation

The notebook is designed for Google Colab.

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn joblib
```

The dataset is downloaded automatically with `kagglehub` (`blastchar/telco-customer-churn`). Local use may require Kaggle credentials.

## Usage

1. Open the notebook in Google Colab (or Jupyter).
2. Run all cells from top to bottom. This downloads the data, trains the model, and writes `customer_churn_model.pkl`.
3. To make predictions with the saved model:

## Key Findings

- **Contract type:** Month-to-month contracts have the highest churn.
- **Internet service:** Fiber optic customers churn more than DSL or no-internet customers.
- **Tenure:** New customers (low tenure) are the highest risk.
- **Pricing:** Higher monthly charges are associated with higher churn.
- **Model performance:** About 80% accuracy on the test set, versus a ~73% majority-class baseline. Recall on the churn class is lower than on the non-churn class, so the decision threshold could be tuned if catching churners matters more than false alarms.
- **Recommendation:** Prioritize retention offers for new, month-to-month, high-bill, fiber customers and encourage longer contracts and bundled security/support add-ons.

## Dependencies

- Python 3.9+
- kagglehub
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
