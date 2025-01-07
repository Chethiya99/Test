__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.utilities import SQLDatabase
import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process, LLM
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Pulse iD - Database Query & Email Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.session_state.email_results = []
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

def get_api_key():
    """Function to get API Key from user input"""
    return st.sidebar.text_input("Enter Your API Key:", type="password")

# Get API Key
api_key = get_api_key()
if api_key:
    st.session_state.api_key = api_key

# Database Path Input
db_path = st.sidebar.text_input("Database Path:", "merchant_data.db")
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])

# Initialize SQL Database and Agent
if db_path and api_key and not st.session_state.db:
    try:
        # Initialize Groq LLM
        llm = ChatGroq(temperature=0, model_name=model_name, api_key=st.session_state.api_key)
        # Initialize SQLDatabase
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
        # Create SQL Agent
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
                    st.session_state.raw_output = result['output'] if isinstance(result, dict) else result
                    
                    # Process raw output using an extraction agent 
                    extractor_llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
                    extractor_agent = Agent(
                        role="Data Extractor",
                        goal="Extract merchants and emails from the raw output.",
                        backstory="You are an expert in extracting structured information from text.",
                        provider="Groq",
                        llm=extractor_llm 
                    )
                    
                    extract_task = Task(
                        description=f"Extract a list of 'merchants' and their 'emails' from the following text:\n\n{st.session_state.raw_output}",
                        agent=extractor_agent,
                        expected_output="A structured list of merchants and their associated email addresses extracted from the given text."
                    )
                    
                    # Crew execution for extraction 
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()
                    # Assuming extraction results are tuples like (name, email)
                    if extraction_results:
                        # Convert results to a list of dictionaries for easier access later on.
                        st.session_state.merchant_data = [{"name": name, "email": email} for name, email in extraction_results]
                    
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")

# Show previous query results even if Generate Emails is clicked 
if st.session_state.raw_output:
    st.markdown("### Query Results:", unsafe_allow_html=True)
    st.write(st.session_state.raw_output)

if isinstance(st.session_state.merchant_data, list):
    st.markdown("### Extracted Merchants:", unsafe_allow_html=True)
    for merchant in st.session_state.merchant_data:
        merchant_info = f"Merchant: {merchant['name']}, Email: {merchant['email']}"
        st.write(merchant_info)

# Email Generator Button 
if isinstance(st.session_state.merchant_data, list) and len(st.session_state.merchant_data) > 0 and st.button("Generate Emails"):
    with st.spinner("Generating emails..."):
        try:
            # Define email generation agent 
            llm_email = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
            email_agent = Agent(
                role="Email Content Generator",
                goal="Generate personalized marketing emails for merchants.",
                backstory="You are a marketing expert named 'Sumit Uttamchandani' of Pulse iD fintech company skilled in crafting professional and engaging emails for merchants.",
                verbose=True,
                allow_delegation=False,
                llm=llm_email 
            )

            # Prepare email generation tasks for each merchant 
            for merchant in st.session_state.merchant_data:
                merchant_name = merchant['name']
                merchant_email = merchant['email']

                task_description = f"""
                Generate professional marketing emails for the following merchant:

                Merchant: {merchant_name}
                Email: {merchant_email}

                Use the below format:

                Subject: Boost Customer Traffic for '{merchant_name}' – No Upfront Cost

                Dear '{merchant_name}',

                Thank you for reaching out! We’re excited about the opportunity to support '{merchant_name}' in driving customer traffic and enhancing engagement through our Pulse iD Marketplace. Here’s how our program can benefit your business:

                - Increased Customer Footfall: Targeted campaigns bring high-value customers to your location.
                - No Upfront Costs: You only fund the discount or offer provided—no hidden fees, no surprises.
                - Flexibility: You’re in control of your offers and can adjust or opt out anytime.

                Next Steps: Complete our quick onboarding form here: [CTA Link – Merchant Onboarding Form]. This will help us set up your account and customize your campaign.

                If you have any questions or would like to explore offer options, feel free to reach out—I’d be happy to assist!

                Kind Regards,
                [Your Name]
                """

                task = Task(
                    description=task_description,
                    agent=email_agent,
                    expected_output=f"Marketing email for {merchant_name}."
                )

                # Crew execution for each merchant 
                crew = Crew(agents=[email_agent], tasks=[task], process=Process.sequential)
                email_result = crew.kickoff()

                # Display results 
                if email_result.raw:
                    email_content_with_image = f"""
                    <div>
                        <h3>Email for {merchant_name}</h3>
                        <div>{email_result.raw}</div>
                    </div>
                    """
                    st.markdown(email_content_with_image, unsafe_allow_html=True)

            # Store all results in session state (optional)
            if isinstance(st.session_state.email_results, list):
                st.session_state.email_results.append(email_result)
        
        except Exception as e:
            st.error(f"Error generating emails: {str(e)}")

# Footer Section 
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Powered by <strong>Pulse iD</strong> | Built with 🐍 Python and Streamlit</div>",
    unsafe_allow_html=True 
)
