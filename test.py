import pandas as pd
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Cache the model initialization and file reading
@st.cache_resource
def load_model():
    """
    Load the LLM model.
    Cached for reuse across multiple queries.
    """
    return Ollama(
        model="hf.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
        request_timeout=30.0
    )

@st.cache_resource
def load_agent(llm):
    """
    Initialize the ReActAgent with cashflow tool and model.
    """
    cashflow_tool = FunctionTool.from_defaults(fn=invokecashflow)
    return ReActAgent.from_tools([cashflow_tool], llm=llm, verbose=False)

@st.cache_data
def read_cashflow_data():
    """
    Read the CashFlow data from the predefined file.
    """
    file_path = "/mnt/data/file-L5yqc7tY2MqNMvS4xbMExR"
    data = pd.read_excel(file_path, sheet_name="CashFlow")
    return data

def invokecashflow():
    """
    Extracts cash flow information from the predefined Excel file.
    """
    try:
        data = read_cashflow_data()
        return data.to_string()
    except Exception as e:
        return f"Error while reading CashFlow data: {str(e)}"

# Streamlit Application
st.title("Interactive Cash Flow Analysis with LLM")

# Load model and agent
llm = load_model()
agent = load_agent(llm)

# Optimized Prompt Integration
optimized_prompts = {
    "Summarize Trends": "Summarize the overall cash flow trends for FY22 and FY23.",
    "Quarterly Breakdown": "Provide a breakdown of cash flow activities for Q1 FY23.",
    "Compare Quarters": "Compare the net cash provided by operating activities between Q1 FY22 and Q1 FY23.",
    "Category Analysis": "What is the total depreciation and amortization expense for FY22?",
    "Trend Analysis": "Identify the quarter with the highest net cash provided by operating activities in FY23.",
}

# User Input
st.write("### Ask Questions About the Cash Flow Data")
query_type = st.selectbox(
    "Select a query type:",
    list(optimized_prompts.keys())
)
user_query = optimized_prompts.get(query_type, "")

# Generate response
if user_query:
    with st.spinner("Processing your query..."):
        cashflow_data = invokecashflow()
        prompt = f"""
        You are an AI financial analyst. Use the following cash flow data to answer the user's question.

        Cash Flow Data:
        {cashflow_data}

        User Query:
        {user_query}
        """
        response = agent.chat(prompt)
    st.success("Response received!")
    st.write(response)

# Optional: Display Raw Cash Flow Data
if st.checkbox("Show Raw Cash Flow Data"):
    try:
        df = read_cashflow_data()
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading cash flow data: {str(e)}")
