import streamlit as st
import pickle
import numpy as np

# -------------------------
# 1. Load models
# -------------------------
with open("Asaad_Salaries_Models.pkl", "rb") as f:
    data = pickle.load(f)

models = data["models"]
month_map = data["month_map"]

# -------------------------
# 2. Prediction function
# -------------------------
def predict(full_name, month, year):
    if full_name not in models:
        return None
    m = models[full_name]
    x = np.array([month_map[month], year])
    x = (x - m["X_mean"]) / m["X_std"]
    x = np.hstack([1, x]).reshape(1,-1)
    y_norm = x @ m["w"]
    y = float(y_norm * m["y_std"] + m["y_mean"])
    return y

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("Salary Predictor")

full_name = st.text_input("Enter Employee's full name :")
month = st.text_input("Enter a valid month :")
year = st.number_input("Enter a valid year :", min_value=2000, max_value=2100, value=2025, step=1)

if st.button("Predict Salary"):
    if not full_name or not month:
        st.warning("Fill all form spots.")
    else:
        month = month.lower()
        if month not in month_map:
            st.error("Invalid Month !")
        else:
            salaire = predict(full_name, month, year)
            if salaire is None:
                st.error("Unknown Employee.")
            else:
                st.success(f"💵 Predicted Salary for {full_name} in {month} {year} : {salaire:.2f} MAD")
