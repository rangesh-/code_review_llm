import os
import pandas as pd
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_huggingface_llm import HuggingFaceLLM  # Replace ChatOpenAI

# Load a Hugging Face model suitable for summarization or conversational tasks
llm = HuggingFaceLLM.from_pretrained("facebook/bart-large-cnn")  # Example: BART for summarization

@tool
def invokecashflow():
    """Cashflow information from Excel, Pertaining to Verizon. Portfolio Includes VCG, VBG, CSG."""
    print('Inside Cashflow Tools')
    data = pd.read_excel("C://Users////Documents//CashFlow.xlsx", sheet_name="CashFlow")
    return data.to_string()

tools = [invokecashflow]

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Cashflow assistant. Do the needful with respect to cashflow/finance statements or queries i.e utilize tools to make decision. Tools that you own is specific to  Information."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Query and get a response
query = "Can you summarize the cashflow statement for ?"
res = agent_executor.invoke({"input": query})
print(res['output'])
