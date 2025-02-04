# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define the schema for structured output
response_schemas = [
    ResponseSchema(name="graph_type", description="Type of graph to plot (e.g., bar, line, scatter, pie)."),
    ResponseSchema(name="x_column", description="Column to use for the x-axis."),
    ResponseSchema(name="y_column", description="Column to use for the y-axis."),
    ResponseSchema(name="names_column", description="Column to use for names in a pie chart."),
    ResponseSchema(name="values_column", description="Column to use for values in a pie chart.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Load and embed Excel data
def load_and_embed_excel(file):
    # Load Excel file
    df = pd.read_excel(file)
    
    # Convert DataFrame to text for embedding
    text_data = df.to_string(index=False)
    
    # Split text into chunks (for better embedding)
    chunks = [text_data[i:i + 1000] for i in range(0, len(text_data), 1000)]
    
    # Embed the text using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return df, vector_store

# Initialize Ollama LLM
def initialize_ollama():
    # Initialize Ollama with the desired model (e.g., llama2)
    llm = Ollama(model="llama3.1:8b")
    return llm

# Create Retrieval QA Chain
def create_qa_chain(llm, vector_store):
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Generate response and decide visualization
def generate_response(qa_chain, query):
    # Get response from the QA chain
    response = qa_chain({"query": query})
    answer = response["result"]
    
    # Decide visualization based on the query
    if "visualize" in query.lower() or "plot" in query.lower():
        # Ask the LLM to decide the graph type and columns
        graph_prompt = (
            f"Based on the data and the query '{query}', decide the type of graph to plot and the columns to use. "
            f"Respond with a structured output containing the graph type, x_column, y_column, names_column, and values_column.\n"
            f"{format_instructions}"
        )
        graph_decision = qa_chain({"query": graph_prompt})["result"]
        print(graph_decision)
        try:
            # Parse the structured output
            graph_decision = output_parser.parse(graph_decision)
            return answer, graph_decision
        except Exception as e:
            st.warning(f"Failed to parse LLM's graph decision: {e}")
            return answer, None
    else:
        return answer, None

# Plot graph based on LLM decision
def plot_graph(df, graph_decision):
    try:
        graph_type = graph_decision.get("graph_type")
        x_column = graph_decision.get("x_column")
        y_column = graph_decision.get("y_column")
        names_column = graph_decision.get("names_column")
        values_column = graph_decision.get("values_column")
        
        if graph_type == "bar":
            if x_column in df.columns and y_column in df.columns:
                fig = px.bar(df, x=x_column, y=y_column, title=f"Bar Chart of {y_column} by {x_column}")
            else:
                st.warning(f"Cannot plot bar chart: Columns '{x_column}' or '{y_column}' not found.")
                return
        elif graph_type == "line":
            if x_column in df.columns and y_column in df.columns:
                fig = px.line(df, x=x_column, y=y_column, title=f"Line Chart of {y_column} over {x_column}")
            else:
                st.warning(f"Cannot plot line chart: Columns '{x_column}' or '{y_column}' not found.")
                return
        elif graph_type == "scatter":
            if x_column in df.columns and y_column in df.columns:
                fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter Plot of {y_column} vs {x_column}")
            else:
                st.warning(f"Cannot plot scatter plot: Columns '{x_column}' or '{y_column}' not found.")
                return
        elif graph_type == "pie":
            if names_column in df.columns and values_column in df.columns:
                fig = px.pie(df, names=names_column, values=values_column, title=f"Pie Chart of {values_column} by {names_column}")
            else:
                st.warning(f"Cannot plot pie chart: Columns '{names_column}' or '{values_column}' not found.")
                return
        else:
            st.warning(f"Unsupported graph type: {graph_type}")
            return
        
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting the graph: {e}")

# Streamlit App
def main():
    st.title("Excel Data Analysis with Ollama and Streamlit")
    
    # Upload Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        # Load and embed Excel data
        df, vector_store = load_and_embed_excel(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        # Initialize Ollama LLM
        llm = initialize_ollama()
        
        # Create QA chain
        qa_chain = create_qa_chain(llm, vector_store)
        
        # User query
        query = st.text_input("Ask a question about the data:")
        
        if query:
            # Generate response and decide visualization
            answer, graph_decision = generate_response(qa_chain, query)
            st.write("Answer:")
            st.write(answer)
            
            # Plot graph if applicable
            if graph_decision:
                st.write(f"LLM decided to plot a {graph_decision.get('graph_type')} chart.")
                plot_graph(df, graph_decision)

# Run the app
if __name__ == "__main__":
    main()