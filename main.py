
import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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


st.title("Resume Screening App")
st.write("Upload a CSV file with resumes and job descriptions to screen candidates.")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)


    st.write("Uploaded Data:")
    st.dataframe(df)


    if 'resume_text' not in df.columns or 'job_description' not in df.columns:
        st.error("The CSV file must contain 'resume_text' and 'job_description' columns.")
    else:

        st.write("Generating embeddings... (This may take a few moments)")
        df['resume_embedding'] = df['resume_text'].apply(get_embeddings)
        df['job_embedding'] = df['job_description'].apply(get_embeddings)


        st.write("Calculating similarity scores...")
        def calculate_similarity(row):
            resume_embedding = np.array(row['resume_embedding'])
            job_embedding = np.array(row['job_embedding'])
            return cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]

        df['similarity'] = df.apply(calculate_similarity, axis=1)


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
