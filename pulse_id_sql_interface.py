__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import re
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process, LLM

# Page Configuration
st.set_page_config(page_title="Pulse iD - Database Query & Email Generator", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'merchant_data' not in st.session_state:
    st.session_state.merchant_data = None
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = ""
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None
if 'email_results' not in st.session_state:
    st.session_state.email_results = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Function to read the email task description from a text file
def read_email_task_description(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

# Set the path to your text file
description_file_path = 'email_descriptions/email_task_description.txt'

# Header Section with Title and Logo
st.image("logo.png", width=150)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Pulse iD - SQL Database Query Interface</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Interact with your merchant database and generate emails with ease!</h4>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Settings")
def get_api_key():
    return st.sidebar.text_input("Enter Your API Key:", type="password")

# Get API Key
api_key = get_api_key()
if api_key:
    st.session_state.api_key = api_key

# Database Path Input
db_path = st.sidebar.text_input("Database Path:", "merchant_data.db")
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])

# Initialize SQL Database and Agent if necessary
if db_path and api_key and not st.session_state.db:
    try:
        llm = ChatGroq(temperature=0, model_name=model_name, api_key=st.session_state.api_key)
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
        st.session_state.agent_executor = create_sql_agent(llm=llm, db=st.session_state.db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        st.sidebar.success("✅ Database and LLM Connected Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Query Input Section
if st.session_state.db:
    user_query = st.text_area("Enter your query:", placeholder="E.g., Show top 10 merchants and their emails.", key="user_query")
    
    if st.button("Run Query", key="run_query"):
        if user_query:
            with st.spinner("Running query..."):
                try:
                    result = st.session_state.agent_executor.invoke(user_query)
                    st.session_state.raw_output = result['output'] if isinstance(result, dict) else result

                    # Process raw output using an extraction agent
                    extractor_llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
                    extractor_agent = Agent(role="Data Extractor", goal="Extract merchants and emails from the raw output.", backstory="You are an expert in extracting structured information from text.", provider="Groq", llm=extractor_llm)

                    extract_task = Task(description=f"Extract a list of 'merchants' and their 'emails', 'image urls' from the following text:\n\n{st.session_state.raw_output}", agent=extractor_agent, expected_output="A structured list of merchants and their associated email addresses extracted from the given text.")
                    
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()
                    st.session_state.extraction_results = extraction_results if extraction_results else ""
                    st.session_state.merchant_data = st.session_state.extraction_results

                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")

    # Show previous query results even if Generate Emails is clicked
    if st.session_state.raw_output:
        st.markdown("### Query Results:", unsafe_allow_html=True)
        st.write(st.session_state.raw_output)
    
    if st.session_state.extraction_results:
        st.markdown("### Extracted Merchants:", unsafe_allow_html=True)
        st.write(st.session_state.extraction_results.raw)

    # Email Generator Button
    if st.session_state.merchant_data and st.button("Generate Emails", key="generate_emails"):
        with st.spinner("Generating emails..."):
            try:
                llm_email = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
                email_agent = Agent(role="Email Content Generator", goal="Generate personalized marketing emails for merchants.", backstory="You are a marketing expert named 'Sumit Uttamchandani' of Pulse iD fintech company skilled in crafting professional and engaging emails for merchants.", verbose=True, allow_delegation=False, llm=llm_email)

                email_task_description = read_email_task_description(description_file_path)
                task = Task(description=email_task_description.format(merchant_data=st.session_state.merchant_data), agent=email_agent, expected_output="Marketing emails for each selected merchant, tailored to their business details.")
                
                crew = Crew(agents=[email_agent], tasks=[task], process=Process.sequential)
                email_results = crew.kickoff()
                st.session_state.email_results = email_results

                # Display results if email_results.raw exists
                if email_results.raw:
                    email_body = email_results.raw

                    def extract_image_url(email_body):
                        url_pattern = r'https?://[^\s]+'
                        urls = re.findall(url_pattern, email_body)
                        return urls[0] if urls else None

                    image_url = extract_image_url(email_body)

                    if image_url:
                        modified_email_body = email_body.replace("Dear", f"Dear,<br><img src='{image_url}' style='max-width: 100%;' />")
                        st.markdown(modified_email_body, unsafe_allow_html=True)
                    else:
                        st.markdown(email_body, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating emails: {str(e)}")

# Footer Section remains constant across interactions.
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 14px;'>Powered by <strong>Pulse iD</strong> | Built with 🐍 Python and Streamlit</div>", unsafe_allow_html=True)
