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
    2. A detailed summary of the changes made to address the vulnerabilities.

    **Important Instructions:**
    - If the vulnerabilities are at the OS level (e.g., in the base image or core packages), replace the base image with a newer, more secure version.
    - If specific packages need to be updated, include the necessary `RUN` commands to update them.
    - Your response MUST be formatted EXACTLY as follows:
    
    === Fixed Dockerfile ===
    <fixed Dockerfile content>

    === Changes Description ===
    <summary of changes>

    - The "=== Fixed Dockerfile ===" section must contain ONLY the fixed Dockerfile content.
    - The "=== Changes Description ===" section must contain ONLY a detailed summary of the changes made.
    - Do not include any additional text, explanations, or notes outside these sections.

    **Example Response for OS-Level Vulnerability:**
    === Fixed Dockerfile ===
    FROM ubuntu:22.04  # Updated base image to fix OS-level vulnerabilities
    RUN apt-get update && apt-get install -y openssl
    ENV http_proxy=http://proxy.example.com:80
    ENV https_proxy=https://proxy.example.com:80

    === Changes Description ===
    - Replaced the base image with `ubuntu:22.04` to fix OS-level vulnerabilities.
    - Updated the `openssl` package to the latest version.
    - Added environment variables for proxy configuration.

    **Example Response for Package-Level Vulnerability:**
    === Fixed Dockerfile ===
    FROM ubuntu:20.04
    RUN apt-get update && apt-get install -y openssl=2.17-326.e17_9.3  # Updated package to fix CVE-2024-2961
    ENV http_proxy=http://proxy.example.com:80
    ENV https_proxy=https://proxy.example.com:80

    === Changes Description ===
    - Updated the `openssl` package to version `2.17-326.e17_9.3` to fix CVE-2024-2961.
    - Added environment variables for proxy configuration.
    """
)
