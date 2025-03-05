import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from response_generator import get_response

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    vector_store.save_local("faiss_index")
    return vector_store

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    
    answer = get_response(context, user_question)
    
    st.write("Question: ", user_question)
    st.write("Response: ", answer)

def main():
    st.set_page_config("PDF Chatterbox")
    st.header("PDF Chatterbox -Interact with your PDF(s)")

    with st.sidebar:
        st.title("File Selection:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processed")

    user_question = st.text_input("Summarize the contents of your PDF(s) or ask a question from the PDF files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()