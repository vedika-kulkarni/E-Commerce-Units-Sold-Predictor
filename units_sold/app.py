import os
import sys
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Configuration & File Candidates
# -------------------------------------------------------------
MODEL_CANDIDATES = ["best_model.pkl", "best_model (1).pkl"]
SCALER_CANDIDATES = ["scaler.pkl", "scaler (1).pkl"]
CSV_CANDIDATES = [
    "Ecommerce_Sales_Prediction_Dataset.csv",
    os.path.expanduser(r"~\Downloads\Ecommerce_Sales_Prediction_Dataset.csv"),
    r"c:\Users\shri\Downloads\Ecommerce_Sales_Prediction_Dataset.csv"
]

FEATURE_COLUMNS = ['Date', 'Product_Category', 'Price', 'Discount', 'Customer_Segment', 'Marketing_Spend']

CATEGORY_MAPPING = {
    "Electronics": 0,
    "Fashion": 1,
    "Home Decor": 2,
    "Sports": 3,
    "Toys": 4
}

SEGMENT_MAPPING = {
    "Occasional": 0,
    "Premium": 1,
    "Regular": 2
}

def load_pkl_file(file_candidates):
    """Attempts to load a pkl artifact using joblib first, then pickle."""
    for filename in file_candidates:
        if os.path.exists(filename):
            try:
                import joblib
                return joblib.load(filename)
            except Exception:
                pass
            try:
                with open(filename, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None

# -------------------------------------------------------------
# Model & Scaler Helper
# -------------------------------------------------------------
def get_scaler_and_model():
    """
    Loads or reconstructs the scaler and regression model.
    Handles raw predictions/array pickle artifacts gracefully.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    # 1. Load or reconstruct Scaler
    scaler = load_pkl_file(SCALER_CANDIDATES)

    if scaler is None or not hasattr(scaler, "transform"):
        scaler = StandardScaler()
        scaler.mean_ = np.array([5.03497500e+02, 2.04250000e+00, 5.03353162e+02, 2.50536125e+01, 9.80000000e-01, 4.96464797e+03])
        scaler.scale_ = np.array([2.89074407e+02, 1.43812160e+00, 2.87340329e+02, 1.45378970e+01, 8.25893456e-01, 2.79652993e+03])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = 6
        scaler.feature_names_in_ = np.array(FEATURE_COLUMNS)

    # 2. Load or reconstruct Model
    loaded_model = load_pkl_file(MODEL_CANDIDATES)
    model = None
    if loaded_model is not None and hasattr(loaded_model, "predict"):
        model = loaded_model

    if model is None:
        csv_path = None
        for p in CSV_CANDIDATES:
            if os.path.exists(p):
                csv_path = p
                break

        if csv_path:
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import LabelEncoder

                df = pd.read_csv(csv_path)
                for col in ['Date', 'Product_Category', 'Customer_Segment']:
                    df[col] = LabelEncoder().fit_transform(df[col])
                
                X = df[FEATURE_COLUMNS]
                y = df['Units_Sold']
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
                
                X_train_scaled = scaler.transform(X_train)
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)

                try:
                    import joblib
                    joblib.dump(model, "best_model.pkl")
                except Exception:
                    with open("best_model.pkl", "wb") as f:
                        pickle.dump(model, f)
            except Exception:
                model = None

        if model is None:
            model = LinearRegression()
            model.coef_ = np.array([0.29952121, -0.18197176, 0.03089362, -0.28285515, 0.28323524, 0.0394878])
            model.intercept_ = 29.78

    return scaler, model

def encode_date(date_obj):
    """Converts a date object to integer offset matching dataset."""
    base_date = datetime.date(2023, 1, 1)
    diff = (date_obj - base_date).days
    return float(np.clip(diff, 0, 1000))

def predict_units_sold(date_val, category_name, price_val, discount_val, segment_name, spend_val, scaler, model):
    """
    Transforms the 6 input features and returns the predicted units sold.
    """
    if isinstance(date_val, (datetime.date, datetime.datetime)):
        encoded_date = encode_date(date_val)
    else:
        try:
            encoded_date = float(date_val)
        except Exception:
            encoded_date = 500.0

    encoded_cat = float(CATEGORY_MAPPING.get(category_name, 0))
    encoded_seg = float(SEGMENT_MAPPING.get(segment_name, 0))

    features_df = pd.DataFrame([[
        encoded_date,
        encoded_cat,
        float(price_val),
        float(discount_val),
        encoded_seg,
        float(spend_val)
    ]], columns=FEATURE_COLUMNS)

    scaled = scaler.transform(features_df)
    prediction = model.predict(scaled)
    return max(0.0, float(prediction[0]))

# -------------------------------------------------------------
# Streamlit Web GUI
# -------------------------------------------------------------
def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="E-Commerce Units Sold Predictor",
        page_icon="📦",
        layout="centered"
    )

    st.title("📦 E-Commerce Units Sold Predictor")
    st.markdown("Predict the number of **Units Sold** based on category, pricing, discounts, marketing spend, and customer segment.")

    scaler, model = get_scaler_and_model()

    st.sidebar.header("About Model")
    st.sidebar.info(
        "**Best Model**: Linear Regression\n\n"
        "**Target**: Units Sold\n\n"
        "**Features**:\n"
        "- Date\n"
        "- Product Category\n"
        "- Price ($)\n"
        "- Discount (%)\n"
        "- Customer Segment\n"
        "- Marketing Spend ($)"
    )

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "Product Category:",
            options=["Electronics", "Fashion", "Home Decor", "Sports", "Toys"],
            index=0
        )
        price = st.number_input(
            "Price ($):",
            min_value=1.0,
            max_value=2000.0,
            value=500.0,
            step=10.0
        )
        discount = st.slider(
            "Discount (%):",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.5
        )

    with col2:
        segment = st.selectbox(
            "Customer Segment:",
            options=["Regular", "Occasional", "Premium"],
            index=0
        )
        spend = st.number_input(
            "Marketing Spend ($):",
            min_value=0.0,
            max_value=20000.0,
            value=5000.0,
            step=100.0
        )
        selected_date = st.date_input(
            "Order Date:",
            value=datetime.date(2023, 6, 15)
        )

    st.markdown("---")
    if st.button("Predict Units Sold", type="primary", use_container_width=True):
        units = predict_units_sold(
            selected_date, category, price, discount, segment, spend, scaler, model
        )
        effective_price = price * (1 - (discount / 100.0))
        est_revenue = units * effective_price

        st.success("### Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Units Sold", f"{round(units)} units", f"{units:.2f} exact")
        with res_col2:
            st.metric("Estimated Revenue", f"${est_revenue:,.2f}")

# -------------------------------------------------------------
# Tkinter Desktop GUI (Fallback or Standalone)
# -------------------------------------------------------------
def run_tkinter():
    import tkinter as tk
    from tkinter import ttk, messagebox

    scaler, model = get_scaler_and_model()

    root = tk.Tk()
    root.title("E-Commerce Units Sold Predictor")
    root.geometry("480x560")
    root.resizable(False, False)
    root.configure(bg="#f4f6f9")

    title_label = tk.Label(
        root, text="Units Sold Predictor",
        font=("Helvetica", 16, "bold"), bg="#f4f6f9", fg="#2c3e50"
    )
    title_label.pack(pady=12)

    model_label = tk.Label(
        root, text="Model: Linear Regression (Best Model)",
        font=("Helvetica", 10, "italic"), bg="#f4f6f9", fg="#7f8c8d"
    )
    model_label.pack(pady=2)

    frame = tk.Frame(root, bg="#ffffff", bd=1, relief="groove", padx=15, pady=15)
    frame.pack(padx=20, pady=10, fill="both", expand=True)

    # Product Category
    tk.Label(frame, text="Product Category:", font=("Helvetica", 10, "bold"), bg="#ffffff").pack(anchor="w", pady=(2, 2))
    cb_category = ttk.Combobox(frame, values=["Electronics", "Fashion", "Home Decor", "Sports", "Toys"], state="readonly")
    cb_category.set("Electronics")
    cb_category.pack(fill="x", pady=(0, 8))

    # Customer Segment
    tk.Label(frame, text="Customer Segment:", font=("Helvetica", 10, "bold"), bg="#ffffff").pack(anchor="w", pady=(2, 2))
    cb_segment = ttk.Combobox(frame, values=["Regular", "Occasional", "Premium"], state="readonly")
    cb_segment.set("Regular")
    cb_segment.pack(fill="x", pady=(0, 8))

    # Price
    tk.Label(frame, text="Price ($):", font=("Helvetica", 10, "bold"), bg="#ffffff").pack(anchor="w", pady=(2, 2))
    entry_price = tk.Entry(frame, font=("Helvetica", 10), justify="left")
    entry_price.insert(0, "500.0")
    entry_price.pack(fill="x", pady=(0, 8))

    # Discount
    tk.Label(frame, text="Discount (%):", font=("Helvetica", 10, "bold"), bg="#ffffff").pack(anchor="w", pady=(2, 2))
    entry_discount = tk.Entry(frame, font=("Helvetica", 10), justify="left")
    entry_discount.insert(0, "25.0")
    entry_discount.pack(fill="x", pady=(0, 8))

    # Marketing Spend
    tk.Label(frame, text="Marketing Spend ($):", font=("Helvetica", 10, "bold"), bg="#ffffff").pack(anchor="w", pady=(2, 2))
    entry_spend = tk.Entry(frame, font=("Helvetica", 10), justify="left")
    entry_spend.insert(0, "5000.0")
    entry_spend.pack(fill="x", pady=(0, 8))

    # Date Index / Day
    tk.Label(frame, text="Day Offset (0 - 1000):", font=("Helvetica", 10, "bold"), bg="#ffffff").pack(anchor="w", pady=(2, 2))
    entry_date = tk.Entry(frame, font=("Helvetica", 10), justify="left")
    entry_date.insert(0, "500")
    entry_date.pack(fill="x", pady=(0, 10))

    res_units = tk.Label(frame, text="Predicted Units Sold: --", font=("Helvetica", 12, "bold"), fg="#27ae60", bg="#ffffff")
    res_rev = tk.Label(frame, text="Estimated Revenue: --", font=("Helvetica", 11), fg="#2980b9", bg="#ffffff")

    def on_predict():
        try:
            p_val = float(entry_price.get())
            d_val = float(entry_discount.get())
            s_val = float(entry_spend.get())
            dt_val = float(entry_date.get())
            cat_val = cb_category.get()
            seg_val = cb_segment.get()

            units = predict_units_sold(dt_val, cat_val, p_val, d_val, seg_val, s_val, scaler, model)
            eff_price = p_val * (1 - (d_val / 100.0))
            rev = units * eff_price

            res_units.config(text=f"Predicted Units: {round(units)} ({units:.2f})")
            res_rev.config(text=f"Estimated Revenue: ${rev:,.2f}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for all numeric fields.")

    btn_predict = tk.Button(
        frame, text="Predict Units Sold", command=on_predict,
        font=("Helvetica", 11, "bold"), bg="#3498db", fg="white",
        activebackground="#2980b9", activeforeground="white", pady=5, relief="flat"
    )
    btn_predict.pack(fill="x", pady=10)

    res_units.pack(pady=2)
    res_rev.pack(pady=2)

    root.mainloop()

# -------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    if "streamlit" in sys.modules or (len(sys.argv) > 1 and sys.argv[1] == "--streamlit"):
        run_streamlit()
    else:
        if "--tkinter" in sys.argv:
            run_tkinter()
        else:
            try:
                import streamlit
                from streamlit.web import cli as stcli
                sys.argv = ["streamlit", "run", sys.argv[0]]
                sys.exit(stcli.main())
            except ImportError:
                run_tkinter()
