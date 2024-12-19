import os
import pandas as pd
from transformers import pipeline
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import BaseLLM  # LangChain-compatible base class

# Define a LangChain-compatible LLM wrapper
class HuggingFaceLLM(BaseLLM):
    def __init__(self, pipeline_func):
        self.pipeline = pipeline_func

    def _call(self, prompt: str, stop=None):
        """Required method for LangChain LLMs to handle text generation."""
        result = self.pipeline(prompt, max_length=150, min_length=40, do_sample=False)
        return result[0]["summary_text"]

    @property
    def _llm_type(self):
        """Return type of LLM."""
        return "custom_huggingface"

# Load the Hugging Face summarization pipeline
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize the LangChain-compatible LLM
llm = HuggingFaceLLM(summarizer_pipeline)

# Define the cashflow tool
@tool
def invokecashflow():
    """Fetches and summarizes cashflow information from an Excel sheet."""
    print("Inside Cashflow Tool")
    file_path = "C://Users//rangesh//Documents//CashFlow.xlsx"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at {file_path}")

    data = pd.read_excel(file_path, sheet_name="CashFlow")
    return data.to_string()

# Define the tools
tools = [invokecashflow]

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Cashflow assistant. Do the needful with respect to cashflow/finance statements or queries i.e utilize tools to make decisions. Tools that you own are specific to Verizon Information."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the LangChain agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Query and get a response
query = "Can you summarize the cashflow statement for Verizon?"
res = agent_executor.invoke({"input": query})
print(res['output'])
