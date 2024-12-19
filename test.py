import os
import pandas as pd
from transformers import pipeline
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Define a custom LLM wrapper for Hugging Face
class HuggingFaceLLM:
    def __init__(self, pipeline_func):
        self.pipeline = pipeline_func

    def bind_tools(self, tools):
        """Required by LangChain to bind tools."""
        self.tools = tools

    def __call__(self, input_text):
        """Call the Hugging Face pipeline to process the input."""
        result = self.pipeline(input_text, max_length=150, min_length=40, do_sample=False)
        return result[0]["summary_text"]

# Load the Hugging Face summarization pipeline
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize the custom LLM
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

tools = [invokecashflow]

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Cashflow assistant. Do the needful with respect to cashflow/finance statements or queries i.e utilize tools to make decision. Tools that you own is specific to Verizon Information."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Bind the tools to the LLM
llm.bind_tools(tools)

# Create the LangChain agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Query and get a response
query = "Can you summarize the cashflow statement for Verizon?"
res = agent_executor.invoke({"input": query})
print(res['output'])
