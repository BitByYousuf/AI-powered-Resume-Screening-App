
import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader


API_TOKEN = "YOUR TOKEN HERE"
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def get_embeddings(text):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


st.title("Resume Screening App")
st.write("Upload resumes in PDF format and enter a job description to screen candidates.")

# Upload PDF files
uploaded_files = st.file_uploader("Upload resumes (PDF format)", type=["pdf"], accept_multiple_files=True)


job_description = st.text_area("Enter the job description:")

if uploaded_files and job_description:

    resumes = []
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        resumes.append({"resume_text": text})

    df = pd.DataFrame(resumes)


    st.write("Extracted Resume Texts:")
    st.dataframe(df)


    st.write("Generating embeddings for resumes... (This may take a few moments)")
    df['resume_embedding'] = df['resume_text'].apply(get_embeddings)


    st.write("Generating embedding for the job description...")
    job_embedding = get_embeddings(job_description)


    st.write("Calculating similarity scores...")
    def calculate_similarity(resume_embedding):
        return cosine_similarity(np.array(resume_embedding).reshape(1, -1), np.array(job_embedding).reshape(1, -1))[0][0]

    df['similarity'] = df['resume_embedding'].apply(calculate_similarity)


    st.write("Top Resumes by Similarity:")
    top_resumes = df.sort_values(by='similarity', ascending=False).head(5)
    st.dataframe(top_resumes[['resume_text', 'similarity']])


    threshold = st.slider("Set a similarity threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)


    filtered_resumes = df[df['similarity'] > threshold]


    st.write(f"Filtered Resumes (Similarity > {threshold}):")
    st.dataframe(filtered_resumes[['resume_text', 'similarity']])


    if st.button("Download Filtered Resumes as CSV"):
        filtered_resumes.to_csv('filtered_resumes.csv', index=False)
        st.success("Filtered resumes saved to 'filtered_resumes.csv'.")
        with open('filtered_resumes.csv', 'rb') as f:
            st.download_button(
                label="Click to Download",
                data=f,
                file_name='filtered_resumes.csv',
                mime='text/csv'
            )
