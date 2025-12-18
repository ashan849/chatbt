import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Simple RAG Chatbot")

st.title("Simple RAG Chatbot")

# Load and split documents
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("data/sample.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectordb

vectordb = load_vectorstore()

llm = OpenAI(temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

query = st.text_input("Ask a question")

if query:
    answer = qa_chain.run(query)
    st.write("Answer:")
    st.write(answer)
