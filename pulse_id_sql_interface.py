__import__('pysqlite3')
import sys
import os
import re
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_community.llms import Ollama

# Page Configuration
st.set_page_config(
    page_title="Pulse iD - Interactive Workflow",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = ""
if 'merchant_data' not in st.session_state:
    st.session_state.merchant_data = None
if 'email_results' not in st.session_state:
    st.session_state.email_results = None

# Sidebar: Settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter API Key:", type="password")
db_path = st.sidebar.text_input("Database Path:", "merchant_data.db")
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])

# Initialize database and agent
if api_key and db_path and not st.session_state.db:
    try:
        llm = ChatGroq(temperature=0, model_name=model_name, api_key=api_key)
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            db=st.session_state.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        st.sidebar.success("✅ Connected Successfully!")
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")

# Header
st.title("🔄 Pulse iD - Repeating Workflow")

# Workflow: Ask Question
if st.session_state.db:
    st.subheader("Step 1: Ask a Question")
    user_query = st.text_area("Enter your SQL-related question:", placeholder="E.g., Show the top 10 merchants.")
    
    if st.button("Run Query"):
        if user_query:
            try:
                with st.spinner("Processing query..."):
                    result = st.session_state.agent_executor.invoke(user_query)
                    st.session_state.raw_output = result['output'] if isinstance(result, dict) else result
                st.success("Query executed successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

# Display Query Results
if st.session_state.raw_output:
    st.subheader("Step 2: Query Results")
    st.write(st.session_state.raw_output)

    # Proceed to Email Generation
    if st.button("Generate Emails"):
        try:
            with st.spinner("Generating emails..."):
                # Simulate email generation logic
                email_output = f"Emails generated based on query results:\n\n{st.session_state.raw_output}"
                st.session_state.email_results = email_output
                st.success("Emails generated successfully!")
        except Exception as e:
            st.error(f"Email generation failed: {e}")

# Display Email Results
if st.session_state.email_results:
    st.subheader("Step 3: Email Results")
    st.text(st.session_state.email_results)

    # Reset for the next cycle
    if st.button("Start Over"):
        st.session_state.raw_output = ""
        st.session_state.email_results = None
        st.success("Workflow reset! You can now ask a new question.")
