import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tiktoken
import pandas as pd  # ✅ Import Pandas for Table Formatting

# ✅ Check Streamlit Version
print(f"Streamlit Version: {st.__version__}")

# ✅ Check if API key is available
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API Key is missing. Set GOOGLE_API_KEY in your environment.")
    st.stop()

# ✅ Initialize AI Model
ai_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.7)

# ✅ Define the AI Prompt for SWOT Analysis
swot_prompt = """
You are an expert business consultant. Given the company information below, provide a **detailed** SWOT Analysis in **structured format only**.

### Strengths  
(Provide at least 3-5 specific strengths related to the company)  

### Weaknesses  
(Provide at least 3-5 specific weaknesses)  

### Opportunities  
(Provide at least 3-5 external opportunities the company can leverage)  

### Threats  
(Provide at least 3-5 threats, including competitors, market risks, etc.)  

**ONLY return the SWOT analysis without additional explanations.**  

Company Details:  
{context}
"""

prompt_template = PromptTemplate(input_variables=["context"], template=swot_prompt)
swot_chain = prompt_template | ai_model  # ✅ New LangChain method

# ✅ Token Counter
encoder = tiktoken.get_encoding("cl100k_base")

# ✅ Function to Generate SWOT
def analyze_swot(input_text):
    return swot_chain.invoke({"context": input_text})

# ✅ Function to Format SWOT into Four Separate Lists
def extract_swot_sections(swot_text):
    """Extracts Strengths, Weaknesses, Opportunities, and Threats as separate lists."""
    lines = swot_text.split("\n")
    strengths, weaknesses, opportunities, threats = [], [], [], []

    category = None
    for line in lines:
        line = line.strip()
        if "Strengths" in line:
            category = strengths
        elif "Weaknesses" in line:
            category = weaknesses
        elif "Opportunities" in line:
            category = opportunities
        elif "Threats" in line:
            category = threats
        elif line.startswith("*"):
            category.append(line[1:].strip())  # Remove * bullet points

    return strengths, weaknesses, opportunities, threats

# ✅ Streamlit Web App UI
st.set_page_config(page_title="SWOT Analysis AI Agent")
st.title("📌 AI-Powered SWOT Analysis App")
st.write("Enter company details to generate a SWOT analysis.")

# ✅ Text Input Box
company_details = st.text_area("Enter company details:")

# Initialize analysis_result to avoid errors
analysis_result = ""

if st.button("Generate SWOT"):
    with st.spinner("Generating analysis..."):
        analysis_result = analyze_swot(company_details)

    # ✅ Extract Clean Text from AI Response
    if isinstance(analysis_result, dict) and "content" in analysis_result:
        swot_text = analysis_result["content"]
    else:
        swot_text = str(analysis_result)  # Convert to string if needed

    # ✅ Remove Extra Metadata and Fix Formatting
    swot_text = swot_text.split("additional_kwargs")[0]  # Remove extra metadata
    swot_text = swot_text.replace("\\n", "\n")  # Fix line breaks
    swot_text = swot_text.strip()  # Remove unnecessary spaces

    # ✅ Display SWOT Analysis with Proper Formatting
    st.subheader("📌 SWOT Analysis")
    st.markdown(swot_text, unsafe_allow_html=False)  # Render Markdown properly

    # ✅ Extract SWOT Data into Lists
    strengths, weaknesses, opportunities, threats = extract_swot_sections(swot_text)

    # ✅ Create a 2x2 Layout (Like the Image)
    st.subheader("📊 SWOT Analysis - Visual Representation")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟦 Strengths")
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.write("- No Strengths Identified")

    with col2:
        st.markdown("### 🟦 Weaknesses")
        if weaknesses:
            for item in weaknesses:
                st.write(f"- {item}")
        else:
            st.write("- No Weaknesses Identified")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🟩 Opportunities")
        if opportunities:
            for item in opportunities:
                st.write(f"- {item}")
        else:
            st.write("- No Opportunities Identified")

    with col4:
        st.markdown("### 🟧 Threats")
        if threats:
            for item in threats:
                st.write(f"- {item}")
        else:
            st.write("- No Threats Identified")

    # ✅ Token tracking
    query_tokens = len(encoder.encode(company_details))
    response_text = str(analysis_result)  # Convert AI response to plain text
    response_tokens = len(encoder.encode(response_text))

    st.sidebar.write(f"Total Tokens: {query_tokens + response_tokens}")
    st.sidebar.write(f"Query Tokens: {query_tokens}")
    st.sidebar.write(f"Response Tokens: {response_tokens}")

    print(f"Tokens used: Query - {query_tokens}, Response - {response_tokens}")
