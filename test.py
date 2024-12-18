import os
import pandas as pd
from transformers import pipeline
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Load a Hugging Face model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@tool
def invokecashflow():
    """Cashflow information from Excel, Pertaining to Verizon. Portfolio Includes VCG, VBG, CSG."""
    print('Inside Cashflow Tools')
    data = pd.read_excel("C://Users//rangesh//Documents//CashFlow.xlsx", sheet_name="CashFlow")
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

# Custom function to use Hugging Face summarization in the workflow
class HuggingFaceLLM:
    def __init__(self, summarizer_pipeline):
        self.summarizer = summarizer_pipeline

    def __call__(self, input_text):
        return self.summarizer(input_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

# Initialize the Hugging Face LLM wrapper
llm = HuggingFaceLLM(summarizer)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Query and get a response
query = "Can you summarize the cashflow statement for Verizon?"
res = agent_executor.invoke({"input": query})
print(res['output'])
