# LoanTap - Loan Default Prediction

## 💰 Overview

This project implements a machine learning solution for predicting loan defaults using Logistic Regression. The analysis focuses on understanding risk factors and building a predictive model to help financial institutions make better lending decisions.

---

### **Business Problem**

- Predict which customers are likely to default on their loans
- Identify key risk factors affecting repayment behavior
- Build a reliable credit risk prediction system

---

### **Analysis Approach**

**Key Insights:**
- 📊 Target variable: Loan default (1 = default, 0 = not default)
- 🔍 Features: Customer demographics, employment status, loan amount
- 🎯 Threshold optimization for different business scenarios

---

### **Methodology Highlights**

| Aspect | Analysis | Result |
|--------|----------|--------|
| Method | Logistic Regression | Binary classification |
| Validation | Cross-validation | Model evaluation |
| Threshold | Prob > 0.5 (default) | Default prediction |

---

### **Technical Implementation**

```python
# Logistic regression for loan default prediction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def predict_default(df):
    X = df.drop(columns=['default'])
    y = df['default']
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Get probability scores
    probs = model.predict_proba(X_test)[:, 1]
```

---

### **Risk Factors**

Identified key predictors of loan default:
- Employment status & income levels
- Loan amount to annual salary ratio
- Age and tenure factors

---

*By Rahul Sharma | Senior Business Analyst transitioning to AI/DS roles*
