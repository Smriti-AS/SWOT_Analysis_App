import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tiktoken  

# ✅ Check if Streamlit is installed
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
You are a business consultant. Analyze the company's information and provide a SWOT Analysis.

**Strengths**
- Example 1
- Example 2

**Weaknesses**
- Example 1
- Example 2

**Opportunities**
- Example 1
- Example 2

**Threats**
- Example 1
- Example 2
"""

prompt_template = PromptTemplate(input_variables=["context"], template=swot_prompt)
swot_chain = prompt_template | ai_model  # ✅ New LangChain method

# ✅ Token Counter
encoder = tiktoken.get_encoding("cl100k_base")

# ✅ Function to Generate SWOT
def analyze_swot(input_text):
    return swot_chain.invoke({"context": input_text})

# ✅ Streamlit Web App UI
st.set_page_config(page_title="SWOT Analysis AI Agent")
st.title("AI-Powered SWOT Analysis App")
st.write("Enter company details to generate a SWOT analysis.")

# ✅ Text Input Box
company_details = st.text_area("Enter company details:")

if st.button("Generate SWOT"):
    with st.spinner("Generating analysis..."):
        analysis_result = analyze_swot(company_details)

        st.subheader("Generated SWOT Analysis:")
        st.write(analysis_result)

        # ✅ Token tracking
        query_tokens = len(encoder.encode(company_details))
        response_tokens = len(encoder.encode(analysis_result))

        st.sidebar.write(f"Total Tokens: {query_tokens + response_tokens}")
        st.sidebar.write(f"Query Tokens: {query_tokens}")
        st.sidebar.write(f"Response Tokens: {response_tokens}")

        print(f"Tokens used: Query - {query_tokens}, Response - {response_tokens}")
