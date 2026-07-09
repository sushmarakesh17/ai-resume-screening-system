import streamlit as st
import pickle
import fitz  # PyMuPDF

# Load saved files
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("AI Resume Screening System")

# Upload PDF
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    # Read PDF
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    resume_text = ""

    # Extract text from all pages
    for page in pdf:
        resume_text += page.get_text()

    st.subheader("Extracted Resume Text")
    st.write(resume_text)

    if st.button("Predict Job Category"):

        # Convert text into numbers
        data = vectorizer.transform([resume_text])

        # Predict
        prediction = model.predict(data)

        # Convert number back to category name
        category = encoder.inverse_transform(prediction)

        st.success(f"Predicted Category: {category[0]}")