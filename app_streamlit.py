import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection ")
st.write("Upload transactions or enter a single transaction to predict **Fraud** vs **Legit**.")

@st.cache_resource
def load_model():
    model_path=os.path.join("models", "knn_model.pkl")
    if not os.path.exists(model_path):
        st.error(" model.pkl not found. Please train and save the model first.")
        return None
    obj = joblib.load(model_path)
    return obj["pipeline"], obj["features"]

loaded = load_model()
if loaded is None:
    st.stop()
pipeline, FEATURES = loaded  


tab1, tab2 = st.tabs(["📤 CSV Upload", "✍️ Single Transaction"])

#  TAB 1: CSV Upload 
with tab1:
    st.subheader("Predict from CSV")
    st.write("CSV must contain columns: `Time, V1..V28, Amount`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            
            df = df[FEATURES]

            preds = pipeline.predict(df)
            proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba(df)[:, 1]
                except Exception:
                    pass

            out = df.copy()
            out["Prediction"] = np.where(preds == 1, "Fraud", "Legit")
            if proba is not None:
                out["Fraud_Probability"] = proba

            st.success("✅ Predictions ready")

            
            fraud_count = (out["Prediction"] == "Fraud").sum()
            legit_count = (out["Prediction"] == "Legit").sum()
            total = len(out)

            st.markdown(
                f"""
                
                - Total Transactions: **{total}**  
                - 🛑 Fraud: **{fraud_count}**  
                - ✅ Legit: **{legit_count}**
                """
            )

            #  Filter
            filter_opt = st.radio("🔍 Filter results:", ["All", "Fraud only", "Legit only"], horizontal=True)
            if filter_opt == "Fraud only":
                filtered = out[out["Prediction"] == "Fraud"]
            elif filter_opt == "Legit only":
                filtered = out[out["Prediction"] == "Legit"]
            else:
                filtered = out

            st.write(f"Showing {len(filtered)} rows out of {len(out)} total")
            st.dataframe(filtered)

            # Download results
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions CSV", 
                data=csv, 
                file_name="predictions.csv", 
                mime="text/csv"
            )


# ============ TAB 2: Single Transaction ============
with tab2:
    st.subheader("Single Transaction Input")

    with st.sidebar:
        st.header("Input Features")
        time_val = st.number_input("Time", value=0.0, step=1.0)
        amount_val = st.number_input("Amount", value=1.0, step=0.1)

        v_vals = {}
        for i in range(1, 29):
            v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1)

   
    row = {
        "Time": time_val,
        **{f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)},
        "Amount": amount_val
    }
    row_df = pd.DataFrame([row])[FEATURES]

    if st.button("Predict"):
        pred = pipeline.predict(row_df)[0]
        proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(row_df)[:, 1][0]
            except Exception:
                pass

        label = "🛑 Fraud" if pred == 1 else "✅ Legit"
        st.markdown(f"### Prediction: {label}")
        if proba is not None:
            st.write(f"**Fraud probability:** {proba:.4f}")
