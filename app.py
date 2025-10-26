import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tempfile
from flatten_json import flatten
import os
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- Optional: seaborn only for style
import seaborn as sns

# Import your medical extraction backend
from report_extract import (
    extract_text_from_pdf,
    extract_attributes_with_gemini,
)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(page_title="🧬 AI Oncology Suite", layout="wide")
st.title("🧬 AI Oncology Suite")
st.write("A unified platform for **Cancer Survival Prediction** and **Medical Report Extraction**.")

# ============================================
# TAB SELECTION
# ============================================
tab1, tab2 = st.tabs(["📈 Cancer Survival Predictor",
                     "📄 Medical Report Extractor"])

# ============================================
# TAB 1: Cancer Survival Predictor
# ============================================
with tab1:
    st.header("📈 Cancer Survival Predictor (Multi-Cancer Model)")
    st.write("Upload tumor gene expression data (CSV) to predict survival risk using pre-trained Cox models for **Breast** and **Lung** cancers.")

    cancer_type = st.selectbox("Select Cancer Type", [
                               "Breast", "Lung"], key="cancer_type")

    model_files = {
        "Breast": {
            "model": "cox_model_brca.pkl",
            "genes": "sig_genes_brca.pkl",
            "data": "training_df_brca.pkl"
        },
        "Lung": {
            "model": "cox_model_luad.pkl",
            "genes": "sig_genes_luad.pkl",
            "data": "training_df_luad.pkl"
        }
    }

    try:
        with open(model_files[cancer_type]["model"], "rb") as f:
            cph = pickle.load(f)

        with open(model_files[cancer_type]["genes"], "rb") as f:
            top_genes = pickle.load(f)

        with open(model_files[cancer_type]["data"], "rb") as f:
            df_training = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Error loading files for {cancer_type}: {e}")
        st.stop()

    median_risk = df_training["risk"].median()

    uploaded_file = st.file_uploader(f"Upload {cancer_type} Expression CSV", type=[
                                     "csv"], key="expr_upload")
    if uploaded_file:
        new_expr = pd.read_csv(uploaded_file, index_col=0)
        new_expr = new_expr.loc[:, ~new_expr.columns.duplicated()]

        # Log2 CPM normalization
        new_expr_log = np.log2(
            (new_expr.div(new_expr.sum(axis=0), axis=1) * 1e6) + 1)

        # Align with model genes
        missing_genes = set(top_genes) - set(new_expr_log.index)
        if missing_genes:
            st.warning(f"{len(missing_genes)} missing genes filled with 0.")
            for g in missing_genes:
                new_expr_log.loc[g] = 0

        X_new = new_expr_log.loc[top_genes].T

        # Predict risk
        risk_scores = cph.predict_partial_hazard(X_new)
        risk_group = np.where(risk_scores >= median_risk,
                              "High Risk", "Low Risk")

        df_results = pd.DataFrame({
            "patient_id": X_new.index,
            "risk_score": risk_scores.values,
            "risk_group": risk_group
        }).set_index("patient_id")

        st.subheader(f"Predicted Risk Groups ({cancer_type})")
        st.dataframe(df_results)

        # Top gene contributions for first patient
        st.subheader("Top Contributing Genes (First Patient)")
        patient_norm = X_new.iloc[[0]]
        gene_contributions = patient_norm.multiply(
            cph.params_.values, axis=1).T.squeeze()
        top20_genes = gene_contributions.abs().sort_values(ascending=False).head(20)

        st.dataframe(top20_genes)

        plt.figure(figsize=(10, 5))
        top20_genes.sort_values().plot(
            kind='barh',
            color=[
                'red' if x > 0 else 'green' for x in gene_contributions[top20_genes.index].sort_values()]
        )
        plt.xlabel("Contribution to Risk Score")
        plt.title(f"Top Gene Contributions ({X_new.index[0]} - {cancer_type})")
        st.pyplot(plt.gcf())

        # Gemini insights
        st.subheader("Gemini AI Gene Insights")
        try:
            # Replace with secure management
            api_key = "AIzaSyBw4zArHam9rrg9crVp7i04crmjQ8zFi5o"
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            gene_list = ", ".join(top20_genes.index.tolist())
            prompt = (
                f"Provide detailed insight on the following genes in {cancer_type} cancer prognosis: {gene_list}. "
                "Include biological function, therapeutic relevance, and biomarker roles."
            )

            with st.spinner("Fetching insights from Gemini..."):
                response = model.generate_content(prompt)
                with st.expander("AI Gene Explanations"):
                    st.markdown(response.text, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gemini AI error: {e}")

# ============================================
# TAB 2: Medical Report Extractor
# ============================================
with tab2:
    st.header("📄 Medical Report Extraction Tool")
    st.write(
        "Upload a PDF medical report. The AI extracts structured data as JSON and Excel.")

    uploaded_pdf = st.file_uploader("📤 Upload Medical Report (PDF)", type=[
                                    "pdf"], key="pdf_upload")

    if uploaded_pdf:
        st.success(f"✅ File uploaded: {uploaded_pdf.name}")

        mode = st.radio("Select Extraction Mode:",
                        ("Extract ALL information", "Extract SPECIFIC attributes"), index=0)

        specific_attributes = []
        if mode == "Extract SPECIFIC attributes":
            attr_input = st.text_input(
                "Enter comma-separated attributes (e.g., tumor_type, ER_status, HER2_status)")
            if attr_input:
                specific_attributes = [a.strip()
                                       for a in attr_input.split(",") if a.strip()]
                st.info(f"Will extract: {', '.join(specific_attributes)}")

        if st.button("🚀 Extract Information", use_container_width=True):
            with st.spinner("Extracting data from PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_pdf.read())
                        temp_path = temp_file.name

                    text_content = extract_text_from_pdf(temp_path)
                    if not text_content:
                        st.error("Could not extract text from PDF.")
                        st.stop()

                    if mode == "Extract SPECIFIC attributes" and specific_attributes:
                        attrs = ", ".join(specific_attributes)
                        prompt = (
                            "You are an AI assistant capable of understanding complex medical reports.\n"
                            "The report may contain masked or anonymized patient data and some random or corrupted characters.\n"
                            f"Extract ONLY the following attributes from the report: {specific_attributes}\n"
                            "Return only valid JSON with keys matching the requested attributes.\n"
                            "Do not include patient names or identifiers.\n"
                            "If an attribute is not found, set it to null.\n"
                            "Your response should be ONLY JSON, no extra commentary.\n\n"
                        )
                    else:
                        prompt = (
                            "You are an AI assistant capable of understanding complex medical reports.\n"
                            "The report may contain masked or anonymized patient data and some random or corrupted characters.\n"
                            "Extract whatever meaningful information you can find in the report and represent it as JSON.\n"
                            "Return only valid JSON with keys describing contents such as symptoms, diagnoses, treatments, labs, observations, dates, or any other useful details.\n"
                            "Do not include patient names or identifiers.\n"
                            "If you find nothing meaningful, return an empty JSON object {}.\n"
                            "Your response should be ONLY JSON, no extra commentary.\n\n"
                        )

                    result = extract_attributes_with_gemini(
                        text_content, prompt)

                    if isinstance(result, dict) and not result.get("error"):
                        result=flatten(result)
                        df = pd.json_normalize(result)
                        st.success("✅ Extraction Complete!")
                        st.json(result)

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as excel_file:
                            df.to_excel(excel_file.name, index=False)
                            with open(excel_file.name, "rb") as f:
                                st.download_button(
                                    label="📥 Download Excel File",
                                    data=f,
                                    file_name=f"{uploaded_pdf.name.split('.')[0]}_extracted.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                        st.dataframe(df, use_container_width=True)
                    else:
                        st.error("⚠️ Error extracting structured data.")
                        st.code(result.get("raw_response",
                                "No response"), language="json")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("---")

