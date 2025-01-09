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
st.set_page_config(
    page_title="Pulse iD - Database Query & Email Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []  # Store previous results as a list of dictionaries
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Function to read the email task description from a text file
def read_email_task_description(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

# Header Section with Title and Logo
st.image("logo.png", width=150)  # Ensure you have your logo in the working directory
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 Pulse iD - SQL Database Query Interface</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #555;'>Interact with your merchant database and generate emails with ease!</h4>",
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter Your API Key:", type="password")
db_path = st.sidebar.text_input("Database Path:", "merchant_data.db")
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])

# Initialize SQL Database and Agent
if db_path and api_key and not st.session_state.db:
    try:
        llm = ChatGroq(temperature=0, model_name=model_name, api_key=api_key)
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            db=st.session_state.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        st.sidebar.success("✅ Database and LLM Connected Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Query Input Section
if st.session_state.db:
    st.markdown("#### Ask questions about your database:", unsafe_allow_html=True)
    user_query = st.text_area("Enter your query:", placeholder="E.g., Show top 10 merchants and their emails.")

    if st.button("Run Query", key="run_query"):
        if user_query:
            with st.spinner("Running query..."):
                try:
                    # Execute the query using the agent
                    result = st.session_state.agent_executor.invoke(user_query)
                    raw_output = result['output'] if isinstance(result, dict) else result
                    
                    # Extract structured information
                    extractor_llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=api_key)
                    extractor_agent = Agent(
                        role="Data Extractor",
                        goal="Extract merchants and emails from the raw output.",
                        backstory="You are an expert in extracting structured information from text.",
                        provider="Groq",
                        llm=extractor_llm
                    )
                    extract_task = Task(
                        description=f"Extract a list of 'merchants' and their 'emails' from the following text:\n\n{raw_output}",
                        agent=extractor_agent
                    )
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()

                    # Store result in history
                    st.session_state.history.append({
                        "query": user_query,
                        "raw_output": raw_output,
                        "extraction_results": extraction_results
                    })
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")

# Display History of Queries and Results
for idx, record in enumerate(st.session_state.history):
    st.markdown(f"### Query {idx + 1}:")
    st.text(f"Query: {record['query']}")
    st.markdown("**Raw Output:**")
    st.write(record['raw_output'])

    if record.get('extraction_results'):
        st.markdown("**Extracted Merchants:**")
        st.write(record['extraction_results'].raw)

    # Email Generator Section
    if st.button(f"Generate Emails for Query {idx + 1}", key=f"generate_emails_{idx}"):
        with st.spinner("Generating emails..."):
            try:
                email_task_description = read_email_task_description('email_descriptions/email_task_description.txt')
                email_agent = Agent(
                    role="Email Content Generator",
                    goal="Generate personalized marketing emails for merchants.",
                    backstory="You are a marketing expert named 'Sumit Uttamchandani' of Pulse iD.",
                    llm=extractor_llm
                )
                task = Task(
                    description=email_task_description.format(merchant_data=record['extraction_results']),
                    agent=email_agent
                )
                crew = Crew(agents=[email_agent], tasks=[task], process=Process.sequential)
                email_results = crew.kickoff()
                st.markdown("**Generated Emails:**")
                st.write(email_results.raw)
            except Exception as e:
                st.error(f"Error generating emails: {str(e)}")

# Footer Section
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Powered by <strong>Pulse iD</strong> | Built with 🐍 Python and Streamlit</div>",
    unsafe_allow_html=True
)
