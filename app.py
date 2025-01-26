import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import tempfile


API_KEY = "your_google_gemini_api_key_here"  


def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    return documents


def split_text_into_chunks(documents, chunk_size=1000, separator="\n"):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(separator=separator, chunk_size=chunk_size)
    text_chunks = []
    for document in documents:
        text_chunks.extend(text_splitter.split_text(document.page_content))
    return text_chunks


def create_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return faiss_db


def search_faiss_index(faiss_db, query, k=5):
    results = faiss_db.similarity_search(query, k=k)
    return results


def generate_answer(api_key, context, query):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response


st.set_page_config(page_title="Chat with Doc", layout="wide")
st.title("Chat with Doc")


st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    
    documents = load_pdf(uploaded_file)
    st.sidebar.success("PDF loaded successfully!")

    chunks = split_text_into_chunks(documents)
    faiss_db = create_faiss_index(chunks)

    st.subheader("Ask Questions about the Document")
    query = st.text_input("Enter your question here:")
    if query:

        results = search_faiss_index(faiss_db, query)
        context = "\n".join([result.page_content for result in results])

        response = generate_answer(API_KEY, context, query)
        st.write("### Answer:")
        st.success(response)
else:
    st.info("Please upload a PDF file to get started.")
