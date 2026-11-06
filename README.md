# 🛒 E-commerce Sales Prediction

A Machine Learning regression project that analyzes e-commerce sales data and predicts the number of units sold based on product pricing, discounts, customer segments, marketing expenditure, and other factors.

## 📌 Project Overview

Sales forecasting helps businesses understand product demand and make better decisions regarding pricing, promotions, and marketing strategies.

This project explores historical e-commerce sales data and compares multiple regression algorithms to identify a suitable model for predicting product units sold.

## ✨ Features

- Exploratory Data Analysis (EDA)
- Missing value and duplicate record handling
- Numerical and categorical feature analysis
- Correlation and covariance analysis
- Data visualization using Matplotlib and Seaborn
- Comparison of multiple regression models
- Model evaluation using regression metrics
- Interactive sales prediction interface

## 📊 Dataset

The dataset contains **1,000 records** and the following seven columns:

| Column | Description |
|---|---|
| Date | Date of the sales record |
| Product_Category | Category of the product |
| Price | Product price |
| Discount | Discount percentage |
| Customer_Segment | Customer classification |
| Marketing_Spend | Marketing expenditure |
| Units_Sold | Number of units sold (target variable) |

**Target variable:** `Units_Sold`

## 🤖 Machine Learning Models

The project compares five regression algorithms:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor

## 📈 Model Evaluation

The models are evaluated using:

- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the target variable's units.
- **R² Score:** Measures how well the model explains variation in sales.

The model comparison is used to identify the best-performing algorithm based on its R² score.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Streamlit | Interactive web application |

## 📂 Project Structure

```text
E-commerce-Sales-Prediction/
│
├── app.py
├── ecommerce_sales_prediction.ipynb
├── Ecommerce_Sales_Prediction_Dataset.csv
├── best_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

*The file structure above is illustrative; adjust it to match the files actually committed to your repository.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd YOUR_PROJECT_FOLDER
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🖥️ Application Screenshots

Add screenshots of your deployed Streamlit application below.

### 1. Dashboard Overview

<img width="1459" height="800" alt="image" src="https://github.com/user-attachments/assets/37142271-3c8e-433e-b6ee-c9488101a508" />

## 🎯 Project Objectives

- Analyze e-commerce sales patterns.
- Understand the relationship between pricing, discounts, and units sold.
- Explore the influence of customer segments and marketing expenditure.
- Build and compare regression models.
- Predict product sales using machine learning.
- Present insights through an interactive application.

## 🚀 Future Enhancements

- Incorporate larger and more diverse sales datasets.
- Improve model performance through hyperparameter tuning.
- Add time-series forecasting.
- Develop advanced sales trend visualizations.
- Integrate real-time sales data.

## 👩‍💻 Author

**Vedika Kulkarni**

B.Tech – Computer Science and Engineering (AI & Data Science)

GitHub: [vedika-kulkarni](https://github.com/vedika-kulkarni)

---

⭐ If you find this project interesting, consider giving the repository a star!
