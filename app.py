import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Portal")

API_URL = "http://127.0.0.1:8000"

st.subheader("Single User Entry")

# ========== FORM INPUTS ==========
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_val = 0 if gender == "Male" else 1

    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior_val = 1 if senior == "Yes" else 0

    partner = st.selectbox("Partner", ["No", "Yes"])
    partner_val = 1 if partner == "Yes" else 0

    tenure = st.number_input("Tenure (in months)", min_value=0.0, format="%.1f")

    phone_services = st.multiselect("Select Services", ["Phone Service", "Internet Service"])
    phoneservice_val = len(phone_services)

    online_service = st.selectbox("Online Services", ["None", "Online Security", "Online Backup", "Tech Support", "Device Protection"])
    onlineservice_val = 0 if online_service == "None" else 1

    streaming = st.selectbox("Streaming Options", ["None", "Streaming Movies", "Streaming TV", "Streaming Music"])
    streaming_val = 0 if streaming == "None" else 1

    contract = st.selectbox("Contract Type", ["Month to month", "One year", "Two year or more"])
    contract_val = {"Month to month": 0, "One year": 1, "Two year or more": 2}[contract]

    monthlycharges = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
    totalcharges = st.number_input("Total Charges", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    with st.spinner("Predicting..."):
        payload = {
            "gender": gender_val,
            "seniorcitizen": senior_val,
            "partner": partner_val,
            "tenure": tenure,
            "phoneservice": phoneservice_val,
            "onlineservice": onlineservice_val,
            "streaming": streaming_val,
            "contract": contract_val,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            if response.status_code == 200:
                st.success("Prediction Complete")
                st.markdown(f"### 🧾 **Churn Prediction:** {'✅ Not Churn' if result['prediction'] == 0 else '⚠️ Churn'}")

                if 'shap' in result:
                    st.subheader("🔍 SHAP Explanation")
                    st.info("This bar chart shows how each feature influenced the prediction. Positive values push towards churn.")
                    shap_df = pd.DataFrame(result['shap'].items(), columns=["Feature", "Impact"])
                    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

                    top_features = shap_df.head(3)
                    st.markdown("#### 🔝 Top Contributing Features:")
                    for i, row in top_features.iterrows():
                        st.markdown(f"- **{row['Feature']}**: {'🔺' if row['Impact'] > 0 else '🔻'} {row['Impact']:.4f}")

                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap_df.plot(kind='barh', x='Feature', y='Impact', ax=ax, color='skyblue')
                    ax.invert_yaxis()
                    ax.set_title("Feature Impact")
                    st.pyplot(fig)
            else:
                st.error(f"Error: {result['error']}")
        except Exception as e:
            st.error(f"Request Failed: {e}")

st.subheader("📁 Upload File (.csv or .pdf)")
file = st.file_uploader("Upload CSV or PDF File", type=["csv", "pdf"])

if file and st.button("Process File"):
    try:
        file_bytes = file.getvalue()
        file_type = file.name.split(".")[-1]

        response = requests.post(f"{API_URL}/batch-predict", files={"file": (file.name, file_bytes)})
        result = response.json()

        if response.status_code == 200:
            st.success(f"{file_type.upper()} Prediction Successful")

            if "predictions" in result:
                st.subheader("📋 Prediction Results")
                predictions_df = pd.DataFrame(result["predictions"], columns=["Prediction"])
                st.dataframe(predictions_df)

                total = len(predictions_df)
                churned = predictions_df["Prediction"].sum()
                churn_rate = churned / total

                # Show metrics and donut in same row
                st.markdown("### 📊 Summary")
                metric1, metric2, metric3, donut_col = st.columns([1, 1, 1, 2])

                with metric1:
                    st.markdown("**Total Users**")
                    st.markdown(f"### {total}")
                with metric2:
                    st.markdown("**Churned Users**")
                    st.markdown(f"### {churned}")
                with metric3:
                    st.markdown("**Churn Rate**")
                    st.markdown(f"### {churn_rate:.2%}")

                with donut_col:
                    fig, ax = plt.subplots(figsize=(3.2, 3.2))
                    labels = ["Churn", "Retain"]
                    sizes = [churned, total - churned]
                    colors = ["crimson", "green"]
                    wedges, texts, autotexts = ax.pie(
                        sizes,
                        labels=labels,
                        colors=colors,
                        startangle=90,
                        autopct='%1.1f%%',
                        wedgeprops={"width": 0.3, "edgecolor": "white"},
                        textprops={"fontsize": 10}
                    )
                    for t in autotexts:
                        t.set_color("black")
                    ax.set(aspect="equal")
                    st.pyplot(fig)

                shap_list = result.get("shap", [])
                if shap_list:
                    all_shap_df = pd.DataFrame(shap_list)
                    mean_shap = all_shap_df.mean().sort_values(key=abs, ascending=False)

                    st.markdown("### 🔍 SHAP Analysis")
                    viz1, viz2 = st.columns(2)

                    with viz1:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        mean_shap.plot(kind='barh', ax=ax, color='steelblue')
                        ax.set_title("Avg SHAP Feature Impact")
                        ax.set_xlabel("SHAP Value")
                        ax.invert_yaxis()
                        st.pyplot(fig)

                    with viz2:
                        st.markdown("#### 🧠 Top 5 Most Influential Features Across Users")
                        for i, (feature, impact) in enumerate(mean_shap.head(5).items(), 1):
                            direction = "🔺 increases" if impact > 0 else "🔻 decreases"
                            st.markdown(f"**{i}. {feature}** — {direction} churn likelihood by {abs(impact):.4f}")

                    st.markdown("---")
                    st.markdown("#### ℹ️ Interpreting SHAP")
                    st.markdown("- 🔵 **Positive SHAP** → contributes to churn")
                    st.markdown("- 🔴 **Negative SHAP** → retains customer")
                    st.markdown("- 📏 **Magnitude** → importance of feature")

                st.download_button("📥 Download CSV Results", data=predictions_df.to_csv(index=False), file_name="churn_predictions.csv", mime="text/csv")

        else:
            st.error(f"Error: {result['error']}")

    except Exception as e:
        st.error(f"Processing failed: {e}")
