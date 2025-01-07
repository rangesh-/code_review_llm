import os
import pandas as pd
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Streamlit page layout
st.set_page_config(layout="wide")
st.title("CashFlow Engine - Ask me Anything related to Cashflow ")

# Custom Excel reader
class ExcelReader(BaseReader):
    def load_data(self, file_path, extra_info=None):
        # Read the Excel file and convert it to a string representation
        df = pd.read_excel(file_path)
        return [Document(text=df.to_string(index=False))]

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Excel File")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Main content
col1, col2 = st.columns(2)

# Processing uploaded file in the left column
with col1:
    st.header("Uploaded File Preview")
    docs = None

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Use the ExcelReader to read the uploaded file
        reader = ExcelReader()
        docs = reader.load_data(temp_file_path)

        st.write("File successfully loaded and processed!")
        df_preview = pd.read_excel(temp_file_path)
        st.dataframe(df_preview)

        # Create and set the embedding model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        Settings.embed_model = embed_model

        # Create index from documents
        index = VectorStoreIndex.from_documents(docs, show_progress=True)

        # LLM configuration
        ollm = Ollama(model="hf.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", request_timeout=3600.0)
        Settings.llm = ollm

        query_engine = index.as_query_engine()
    else:
        st.write("Please upload an Excel file to proceed.")

# Query and response in the right column
with col2:
    st.header("Query and Response")
    if docs:
        user_query = st.text_input("Enter your query:")

        if user_query:
            with st.spinner("Processing your query..."):
                response = query_engine.query(user_query)
                response_text = response.response  # Extract only the response text
            st.success("Response received!")
            st.write(response_text)
    else:
        st.write("Upload a file in the left panel to enable querying.")
