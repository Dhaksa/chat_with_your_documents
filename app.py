import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from docx import Document
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

def process_documents(uploaded_file):
    extracted_text = ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(temp_path)
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
        elif uploaded_file.name.endswith('.docx'):
            extracted_text = extract_text_from_docx(temp_path)
        elif uploaded_file.name.endswith('.doc'):
            extracted_text = extract_text_from_doc(temp_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Error during text extraction: {e}")
        return None
    
    return extracted_text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_doc(doc_path):
    try:
        docx_path = doc_path + ".docx"
        subprocess.run(["soffice", "--headless", "--convert-to", "docx", doc_path], check=True)
        return extract_text_from_docx(docx_path)
    except Exception as e:
        st.error(f"Error converting DOC to DOCX: {e}")
        return ""

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def rag_pipeline(vectorstore):
    llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key_here"
    st.set_page_config(page_title="Chat with Doc", layout="wide")
    st.title("Chat with Your Documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.sidebar.file_uploader("Upload a PDF, DOC, or DOCX file", type=["pdf", "doc", "docx"])
    if uploaded_file and st.sidebar.button("Process Document"):
        with st.spinner("Processing document..."):
            extracted_text = process_documents(uploaded_file)
            if extracted_text:
                chunks = get_chunks(extracted_text)
                vectorstore = get_embeddings(chunks)
                st.session_state.conversation = rag_pipeline(vectorstore)
                st.sidebar.success("Document processed successfully!")

    st.subheader("Ask Questions about the Document")
    user_question = st.text_input("Enter your question here:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history.append({"question": user_question, "answer": response["answer"]})
        st.write("### Answer:")
        st.success(response["answer"])
    
    st.subheader("Query History")
    for entry in st.session_state.chat_history:
        st.write(f"**Q:** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")

if __name__ == "__main__":
    main()
