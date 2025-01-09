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
    st.session_state.history = []
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Helper Functions
def get_api_key():
    """Prompt user for API key in the sidebar."""
    return st.sidebar.text_input("Enter Your API Key:", type="password")

def read_email_task_description(file_path):
    """Read the email task description from a text file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

# Sidebar Configuration
st.sidebar.header("Settings")
api_key = get_api_key()
if api_key:
    st.session_state.api_key = api_key

db_path = st.sidebar.text_input("Database Path:", "merchant_data.db")
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])

# Database and Agent Initialization
if db_path and api_key and not st.session_state.db:
    try:
        llm = ChatGroq(temperature=0, model_name=model_name, api_key=st.session_state.api_key)
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

# Main Section
st.image("logo.png", width=150)
st.markdown("<h1 style='text-align: center;'>📊 Pulse iD - SQL Database Query Interface</h1>", unsafe_allow_html=True)

if st.session_state.db:
    st.markdown("### Ask questions about your database:")
    user_query = st.text_area("Enter your query:", placeholder="E.g., Show top 10 merchants and their emails.")

    if st.button("Run Query"):
        if user_query:
            try:
                result = st.session_state.agent_executor.invoke(user_query)
                raw_output = result['output'] if isinstance(result, dict) else result

                # Extract merchant data
                extractor_llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
                extractor_agent = Agent(
                    role="Data Extractor",
                    goal="Extract merchants and emails from the raw output.",
                    provider="Groq",
                    llm=extractor_llm
                )
                extraction_task = Task(
                    description=f"Extract merchants and emails from:\n\n{raw_output}",
                    agent=extractor_agent,
                    expected_output="A structured list of merchants and emails."
                )
                extraction_crew = Crew(agents=[extractor_agent], tasks=[extraction_task], process=Process.sequential)
                extraction_results = extraction_crew.kickoff()

                # Save to history
                st.session_state.history.append({
                    "query": user_query,
                    "raw_output": raw_output,
                    "extraction": extraction_results.raw if extraction_results else ""
                })
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("Please enter a query.")

# Display History
for idx, interaction in enumerate(st.session_state.history):
    st.markdown(f"### Interaction #{idx + 1}")
    st.markdown("**Query:**")
    st.write(interaction["query"])
    st.markdown("**Query Result:**")
    st.write(interaction["raw_output"])
    st.markdown("**Extracted Merchants:**")
    st.write(interaction["extraction"])

# Email Generation
if st.button("Generate Emails"):
    if st.session_state.history:
        last_interaction = st.session_state.history[-1]
        merchant_data = last_interaction.get("extraction", "")

        try:
            email_task_description = read_email_task_description("email_descriptions/email_task_description.txt")
            llm_email = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
            email_agent = Agent(
                role="Email Generator",
                goal="Generate emails for merchants.",
                llm=llm_email
            )
            email_task = Task(
                description=email_task_description.format(merchant_data=merchant_data),
                agent=email_agent,
                expected_output="Personalized emails for merchants."
            )
            email_crew = Crew(agents=[email_agent], tasks=[email_task], process=Process.sequential)
            email_results = email_crew.kickoff()
            
            if email_results.raw:
                st.markdown("### Generated Emails:")
                st.markdown(email_results.raw, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating emails: {str(e)}")
