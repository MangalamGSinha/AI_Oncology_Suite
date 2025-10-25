# 🧬 AI Oncology Suite

An integrated **Streamlit** application for:

1. **Cancer Survival Prediction** using pre-trained Cox proportional hazard models (BRCA and LUAD).
2. **Medical Report Extraction** powered by **Google Gemini AI** for structured data extraction from PDF reports.

---

### 🌟 Features

#### 📈 Cancer Survival Predictor

* Upload tumor gene expression data (CSV format).
* Predict survival risk using **Cox proportional hazard models** trained on TCGA data.
* Supports **BRCA** (Breast Cancer) and **LUAD** (Lung Adenocarcinoma).
* Visualize top contributing genes per patient.
* Get **AI-generated biological insights** for top genes via Gemini AI.

#### 📄 Medical Report Extractor

* Upload  **PDF medical reports** .
* Extracts structured information as  **JSON and Excel** .
* Choose between:
  * **Extract ALL information** (comprehensive mode).
  * **Extract SPECIFIC attributes** (targeted mode).
* Supports custom attribute prompts (e.g.,  *tumor_type* ,  *ER_status* ,  *HER2_status* ).

---

### 🏗️ Project Structure

```
AI-Oncology-Suite/
│
├── app.py                        # Main Streamlit application
├── report_extract.py              # Backend for PDF text extraction & Gemini interaction
│
├── cox_model_brca.pkl            # Pre-trained BRCA Cox model
├── sig_genes_brca.pkl            # Signature genes for BRCA model
├── training_df_brca.pkl          # Training dataset (BRCA)
│
├── cox_model_luad.pkl            # Pre-trained LUAD Cox model
├── sig_genes_luad.pkl            # Signature genes for LUAD model
├── training_df_luad.pkl          # Training dataset (LUAD)
│
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

---

### ⚙️ Installation

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Oncology-Suite.git
cd AI-Oncology-Suite
```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🚀 Run the App

```bash
streamlit run app.py
```

Then open the local URL (usually `http://localhost:8501`) in your browser.

---

### 🧩 How It Works

#### 1️⃣ Cancer Survival Predictor

* Normalizes uploaded expression data to  **log2 CPM** .
* Aligns genes with the model’s training features.
* Computes partial hazard risk using the  **CoxPH model** .
* Classifies patients into **High-Risk** or **Low-Risk** based on training median risk.
* Displays top contributing genes and requests biological insights via  **Gemini AI** .

#### 2️⃣ Medical Report Extractor

* Extracts text from uploaded PDFs using `extract_text_from_pdf()`.
* Builds a natural-language prompt for Gemini based on user selection.
* Receives structured JSON and converts it into a clean  **DataFrame** .
* Exports results to **Excel** for download.

---

### 💡 Future Enhancements

* Support for more cancer types (COAD, GBM, etc.)
* Integration with RNA-seq preprocessing modules
* Support for clinical text summarization
* Secure API key management via secrets manager

---

