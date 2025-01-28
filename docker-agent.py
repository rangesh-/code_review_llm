import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

# Title of the app
st.title("Dockerfile Vulnerability Fixer with Sysdig Reports")
st.write("Upload your Dockerfile and Sysdig vulnerability report to get a fixed version!")

# Load Ollama with Llama 2
@st.cache_resource
def load_ollama_model():
    return Ollama(base_url="http://localhost:11434", model="hf.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M")

llm = load_ollama_model()

# Define the prompt template for Sysdig vulnerabilities
prompt_template = PromptTemplate(
    input_variables=["dockerfile", "vulnerabilities"],
    template="""
    You are a Docker security expert. Below is a Dockerfile and a Sysdig vulnerability report:
    
    Dockerfile:
    {dockerfile}
    
    Sysdig Vulnerability Report:
    {vulnerabilities}
    
    Analyze the vulnerabilities and provide:
    1. A fixed, secure, and optimized version of the Dockerfile.
    2. A summary of the changes made to address the vulnerabilities.
    
    Format your response as follows:
    
    === Fixed Dockerfile ===
    <fixed Dockerfile content>
    
    === Changes Description ===
    <summary of changes>
    """
)

# Create the LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit input fields
dockerfile = st.text_area("Paste your Dockerfile here:", height=200)
vulnerabilities = st.text_area("Paste the Sysdig vulnerability report here:", height=100)

# Fix button
if st.button("Fix Dockerfile"):
    if dockerfile and vulnerabilities:
        with st.spinner("Analyzing and fixing vulnerabilities..."):
            # Run the chain
            response = chain.run({
                "dockerfile": dockerfile,
                "vulnerabilities": vulnerabilities
            })
        
        # Parse the response into Fixed Dockerfile and Changes Description
        if "=== Fixed Dockerfile ===" in response and "=== Changes Description ===" in response:
            fixed_dockerfile = response.split("=== Fixed Dockerfile ===")[1].split("=== Changes Description ===")[0].strip()
            changes_description = response.split("=== Changes Description ===")[1].strip()
            
            # Display the Fixed Dockerfile
            st.success("Fixed Dockerfile:")
            st.code(fixed_dockerfile, language="dockerfile")
            
            # Display the Changes Description
            st.success("Changes Description:")
            st.write(changes_description)
        else:
            st.error("The LLM response format is invalid. Please check the prompt template.")
    else:
        st.error("Please provide both a Dockerfile and Sysdig vulnerability report.")
