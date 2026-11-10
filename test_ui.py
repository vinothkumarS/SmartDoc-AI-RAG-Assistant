
import streamlit as st

st.set_page_config(
    page_title="SmartDoc AI Test",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SmartDoc AI")

st.success("Streamlit UI is working successfully!")

st.write("Welcome to my Generative AI RAG Project")

question = st.text_input("Ask your question")

if st.button("Submit"):
    st.write("Your Question:", question)
