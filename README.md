# AI-powered-Resume-Screening-App

This project is an AI/ML-powered **resume screening tool** built using **Streamlit**, designed to automate and improve the candidate shortlisting process. Recruiters can either upload individual PDF resumes along with a job description or provide a CSV containing resume–job description pairs to find the best matching candidates using semantic similarity.

---

## 🚀 Features

- 📄 **PDF Resume Screening** — Upload multiple resumes and match them to a job description.
- 🧾 **CSV Mode** — Upload a CSV with `resume_text` and `job_description` columns for bulk screening.
- 🤖 **AI-Powered Similarity Scoring** — Uses Hugging Face Transformers for semantic comparison.
- 📊 **Similarity Thresholding** — Adjustable threshold to filter top candidates.
- 📥 **Download Filtered Results** — Export shortlisted resumes as CSV.

---

## 🧠 How It Works (AI/ML Overview)

This app uses **natural language processing (NLP)** and **semantic similarity** techniques:

- **Sentence Embeddings**: 
  - Leveraging the `all-MiniLM-L6-v2` model from `sentence-transformers`, each resume and job description is converted into a high-dimensional vector representation.
  - This process captures semantic meaning, not just keywords.

- **Cosine Similarity**:
  - Measures how similar two embedding vectors are.
  - A score close to 1 means high similarity; close to 0 means low similarity.

By comparing these embeddings, the app ranks resumes based on how closely they match the job description in **meaning**.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/resume-screening-app.git
   cd resume-screening-app
