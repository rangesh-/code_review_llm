import os
import pandas as pd
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Function to read cash flow data
def invokecashflow():
    """
    Extracts cash flow information from a predefined Excel file.
    The Excel file should have a sheet named 'CashFlow'.
    """
    try:
        file_path = "C://Users//rangesh//Documents//CashFlow.xlsx"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        st.info("Reading Cashflow data from Excel...")
        data = pd.read_excel(file_path, sheet_name="CashFlow")
        return data.to_string()
    except Exception as e:
        return f"Error while reading CashFlow data: {str(e)}"

# Initialize the FunctionTool
cashflow_tool = FunctionTool.from_defaults(fn=invokecashflow)

# Configure the LLM model
llm = Ollama(
    model="hf.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", 
    request_timeout=180.0
)

# Create the ReActAgent
agent = ReActAgent.from_tools([cashflow_tool], llm=llm, verbose=True)

# Streamlit Application
st.title("Interactive Cash Flow Analysis with LLM")

# User Input
user_query = st.text_input("Ask a question about the cash flow data:")

if user_query:
    with st.spinner("Processing your request..."):
        response = agent.chat(user_query)
    st.success("Response received!")
    st.write(response)

# Display Cash Flow Data (Optional)
if st.checkbox("Show Cash Flow Data"):
    try:
        df = pd.read_excel("C://Users//rangesh//Documents//CashFlow.xlsx", sheet_name="CashFlow")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading cash flow data: {str(e)}")
