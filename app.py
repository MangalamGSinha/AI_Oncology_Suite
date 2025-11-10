import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tempfile
from flatten_json import flatten
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
from dotenv import load_dotenv

#load_dotenv()

api_key = os.getenv("GENAI_API_KEY")

# --- Optional: seaborn only for style
import seaborn as sns

# Import your medical extraction backend
from final_program import (
    extract_text_from_pdf,
    process_single_uploaded_file,
    process_pdf_list,
    generate_clinical_summary
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
selected_tab = st.sidebar.radio("Select Tool",
    ["📈 Cancer Survival Predictor", "📄 Medical Report Extractor"])

if selected_tab == "📈 Cancer Survival Predictor":
    # Show predictor UI and no extraction sidebar
# ============================================
# TAB 1: Cancer Survival Predictor
# =====================
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

        # Fix the original top 20 genes and their order
        top20_genes = gene_contributions.abs().sort_values(ascending=False).head(20)
        fixed_top20_genes = top20_genes.index.tolist()

        st.dataframe(top20_genes)

        # Initial plot with fixed genes and their contributions
        fig, ax = plt.subplots(figsize=(10, 5))
        top20_genes.sort_values().plot(
            kind='barh',
            color=[
                'red' if gene_contributions[gene] > 0 else 'green' for gene in top20_genes.index.sort_values()
            ],
            ax=ax
        )
        ax.set_xlabel("Contribution to Risk Score")
        ax.set_title(f"Top Gene Contributions ({X_new.index[0]} - {cancer_type})")
        st.pyplot(fig)


        # Interactive sliders to adjust gene expressions
        st.subheader("Adjust Gene Expression Levels and Recalculate Risk")

        adjusted_gene_expr = {}
        for gene in fixed_top20_genes:
            initial_val = float(patient_norm[gene].iloc[0])
            slider_val = st.slider(
                label=f"Expression of {gene}",
                min_value=0.0,
                max_value=2.0 * initial_val,
                value=initial_val,
                step=0.01
            )
            adjusted_gene_expr[gene] = slider_val


        # Create full expression profile including unchanged genes
        complete_expr_full = X_new.loc[[X_new.index[0]], top_genes].copy()
        for gene, val in adjusted_gene_expr.items():
            if gene in complete_expr_full.columns:
                complete_expr_full[gene] = val

        # Recalculate risk score using updated expression
        new_risk_score = cph.predict_partial_hazard(complete_expr_full)
        new_risk_group = "High Risk" if new_risk_score.values[0] >= median_risk else "Low Risk"

        st.subheader("Recomputed Risk Prediction")
        st.write({
            "Risk Score": new_risk_score.values[0],
            "Risk Group": new_risk_group
        })

        # Update the existing plot (same top 20 genes, updated contributions)
        
        #updated_gene_contributions = complete_expr_full.iloc[0].multiply(cph.params_.values, axis=0)
        #updated_top20_contribs = updated_gene_contributions.loc[fixed_top20_genes]

        #fig, ax = plt.subplots(figsize=(10, 5))
        #updated_top20_contribs.sort_values().plot(
            #kind='barh',
            #color=['red' if x > 0 else 'green' for x in updated_top20_contribs.sort_values()],
            #ax=ax
        #)
        #ax.set_xlabel("Contribution to Risk Score")
        #ax.set_title(f"Updated Gene Contributions ({X_new.index[0]} - {cancer_type})")
        #st.pyplot(fig)'''


        st.subheader("Gemini AI Gene Insights")
        try:
            # Replace with secure management
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
elif selected_tab == "📄 Medical Report Extractor":
    st.markdown(
        """
        <style>
        #tab2-container .main {
            background: linear-gradient(120deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #1e1e1e;
        }
        #tab2-container .stButton>button {
            background-color: #0078d7;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
        }
        #tab2-container .stButton>button:hover {
            background-color: #005fa3;
            color: #fff;
            transform: scale(1.03);
        }
        #tab2-container .title {
            font-size: 40px;
            text-align: center;
            color: #004080;
            font-weight: bold;
            margin-bottom: 10px;
        }
        #tab2-container .sub {
            font-size: 18px;
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        #tab2-container .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        #tab2-container .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: bold;
            color: #004080;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Wrap all Tab 2 content in a div with id="tab2-container"
    st.markdown('<div id="tab2-container">', unsafe_allow_html=True)


    st.header("📄 Medical Report Extraction Tool")
    # ================================================================
    # 🧠 APP HEADER
    # ================================================================
    st.markdown("<div class='title'>🧬 Medical Report Extraction Tool</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Leveraging AI for structured data extraction from medical PDFs</div>", unsafe_allow_html=True)

    # ================================================================
    # ⚙️ EXTRACTION CONFIGURATION
    # ================================================================
    st.sidebar.markdown("### ⚙️ Extraction Configuration")
    mode = st.sidebar.radio(
        "Select what you want to extract:",
        ("Extract ALL information", "Extract SPECIFIC attributes"),
        index=0,
        help="Choose 'Specific attributes' to target only selected medical parameters."
    )

    specific_attributes = []
    if mode == "Extract SPECIFIC attributes":
        attr_input = st.sidebar.text_input(
            "🧾 Enter comma-separated attributes to extract",
            placeholder="e.g., tumor_type, ER_status, HER2_status"
        )
        if attr_input:
            specific_attributes = [a.strip() for a in attr_input.split(",") if a.strip()]
        else:
            st.sidebar.warning("⚠️ No attributes entered.")

    # Advanced Mode
    advanced_mode = st.sidebar.checkbox("🔬 Advanced Accuracy Mode (Chunking + Merging)", value=True)
    st.sidebar.markdown("---")

    # ================================================================
    # 🧠 BUILD PROMPT (Like in Backend)
    # ================================================================
    if mode == "Extract SPECIFIC attributes" and specific_attributes:
        attributes_list = ", ".join(specific_attributes)
        prompt = (
            "You are an AI assistant capable of understanding complex medical reports.\n"
            "The report may contain masked or anonymized patient data and some random or corrupted characters.\n"
            f"Extract ONLY the following attributes from the report: {attributes_list}\n"
            "Return only valid JSON with keys matching the requested attributes.\n"
            "Do not include patient names or identifiers.\n"
            "If an attribute is not found, omit it or set it to null.\n"
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
    subtab1, subtab2 = st.tabs(["🚀 Single/Batch Extraction", "⚖️ Report Comparison Mode"])

    # ================================================================
    # TAB 1: SINGLE/BATCH EXTRACTION
    # ================================================================
    with subtab1:
        st.markdown("### 📥 Upload Reports for Extraction")
        uploaded_files = st.file_uploader("📄 Upload one or more medical report PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            st.info(f"✅ {len(uploaded_files)} file(s) uploaded.")
            
            # Determine if it's a single or batch job
            is_batch = len(uploaded_files) > 1

            if st.button(f"🚀 Extract Data from {'Batch' if is_batch else 'Single Report'}", use_container_width=True, key="extract_batch_button"):
                with st.spinner("Processing files... This may take a few seconds per report ⏳"):
                    
                    # --- PROCESS FILES ---
                    extracted_data_list = process_pdf_list(uploaded_files, prompt, specific_attributes, advanced_mode)
                    
                    if not extracted_data_list:
                        st.error("❌ Failed to extract any valid data from the uploaded files.")
                        st.stop()

                    # --- CONVERT TO DATAFRAME ---
                    final_dfs = []
                    for result in extracted_data_list:
                        # Convert all nested dicts/lists into strings for Excel compatibility
                        clean_result = {
                            k: (json.dumps(v, ensure_ascii=False, indent=2) if isinstance(v, (dict, list)) else v)
                            for k, v in result.items()
                        }
                        df = pd.json_normalize(clean_result)
                        final_dfs.append(df)

                    # Combine all DataFrames for batch output
                    combined_df = pd.concat(final_dfs, ignore_index=True)

                    st.success(f"✅ Data extraction complete from {len(extracted_data_list)} reports!")
                    if is_batch:
                        st.markdown("### 🧠 AI Batch Clinical Summary")
                        batch_summary_input = {"batch_reports": extracted_data_list}
                        batch_summary = generate_clinical_summary(batch_summary_input, mode="single")
                        st.write(batch_summary)

                    # --- DISPLAY AND DOWNLOAD BATCH/SINGLE ---
                    st.markdown("### 📊 Extracted Information (Table View)")
                    st.dataframe(combined_df, use_container_width=True)

                    # Excel download
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as excel_file:
                        # Flatten the dataframe columns if necessary (json_normalize already does a good job)
                        combined_df.to_excel(excel_file.name, index=False)
                        with open(excel_file.name, "rb") as f:
                            download_filename = "batch_extracted_data.xlsx" if is_batch else f"{uploaded_files[0].name.split('.')[0]}_attributes.xlsx"
                            st.download_button(
                                label="📥 Download Extracted Data as Excel",
                                data=f,
                                file_name=download_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    # Show individual JSON results for transparency
                    if not is_batch:
                        st.markdown("### 📜 Raw JSON Output")
                        st.json(extracted_data_list[0])
                        st.markdown("### 🧠 AI Clinical Narrative Summary")
                        summary = generate_clinical_summary(extracted_data_list[0], mode="single")
                        st.write(summary)

                        # Save JSON + Excel
                        df = pd.json_normalize(extracted_data_list[0])
                        excel_path = "extracted_data.xlsx"
                        df.to_excel(excel_path, index=False)
                        with open("extracted_data.json", "w") as f:
                            json.dump(extracted_data_list[0], f, indent=4)

                        st.download_button("📥 Download Excel", open(excel_path, "rb"), file_name="extracted_data.xlsx")
                        st.download_button("📥 Download JSON", open("extracted_data.json", "rb"), file_name="extracted_data.json")
    with subtab2:
        st.markdown("### ⚖️ Compare Two Medical Reports (e.g., Before/After Treatment)")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            report_a = st.file_uploader("Upload **Report A (e.g., Baseline)**", type=["pdf"], key="report_a_upload")
        with col_b:
            report_b = st.file_uploader("Upload **Report B (e.g., Follow-up)**", type=["pdf"], key="report_b_upload")

        if report_a and report_b:
            st.info("Both files uploaded. Click 'Compare' to proceed.")
            
            if st.button("⚖️ Compare Reports", use_container_width=True, key="compare_button"):
                with st.spinner("Extracting and comparing data... ⏳"):
                    
                    # --- PROCESS REPORT A ---
                    result_a = process_single_uploaded_file(report_a, prompt, specific_attributes, advanced_mode)
                    
                    # --- PROCESS REPORT B ---
                    result_b = process_single_uploaded_file(report_b, prompt, specific_attributes, advanced_mode)
                    
                    if result_a.get("error") or result_b.get("error"):
                        st.error("❌ Failed to extract data from one or both reports.")
                        if result_a.get("error"): st.error(f"Report A Error: {result_a['error']}")
                        if result_b.get("error"): st.error(f"Report B Error: {result_b['error']}")
                        st.stop()
                    
                    st.success("✅ Extraction complete. Displaying comparison.")
                    
                    # --- GENERATE COMPARISON TABLE ---
                    
                    # Get all unique keys from both results
                    all_keys = sorted(list(set(result_a.keys()) | set(result_b.keys())))
                    
                    # Create the comparison data list
                    comparison_data = []
                    for key in all_keys:
                        value_a = result_a.get(key, "N/A")
                        value_b = result_b.get(key, "N/A")
                        
                        # Convert complex types to JSON string for comparison clarity in table
                        value_a_str = json.dumps(value_a, ensure_ascii=False) if isinstance(value_a, (dict, list)) else value_a
                        value_b_str = json.dumps(value_b, ensure_ascii=False) if isinstance(value_b, (dict, list)) else value_b
                        
                        # Determine if the values are different
                        status = "✅ Same"
                        if key not in ['source_file', 'error'] and value_a_str != value_b_str:
                            status = "⚠️ Different"
                            
                        comparison_data.append({
                            "Attribute": key,
                            report_a.name: value_a_str,
                            report_b.name: value_b_str,
                            "Status": status
                        })
                        
                    df_compare = pd.DataFrame(comparison_data)

                    st.markdown("### 🔍 Side-by-Side Attribute Comparison")
                    st.dataframe(df_compare, use_container_width=True, hide_index=True)

                    # 🧬 Comparative Clinical Summary
                    st.markdown("### 🧬 AI Comparative Clinical Summary")
                    comparison_input = {"Report_A": result_a, "Report_B": result_b}
                    comparison_summary = generate_clinical_summary(comparison_input, mode="comparison")
                    st.write(comparison_summary)
                    
                    # Download button for the comparison table
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as excel_file:
                        df_compare.to_excel(excel_file.name, index=False)
                        with open(excel_file.name, "rb") as f:
                            st.download_button(
                                label="📥 Download Comparison Table (Excel)",
                                data=f,
                                file_name=f"Comparison_{report_a.name.split('.')[0]}_vs_{report_b.name.split('.')[0]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )    
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "",
    unsafe_allow_html=True
)
