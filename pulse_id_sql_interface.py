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
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []  # Store all interactions (queries, results, emails)
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = "merchant_data_singapore.db"  # Default database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False  # Track if the database is initialized
if 'selected_template' not in st.session_state:
    st.session_state.selected_template = "email_task_description1.txt"  # Default template
if 'trigger_rerun' not in st.session_state:
    st.session_state.trigger_rerun = False  # Track if a re-run is needed

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

def get_api_key():
    """Function to get API Key from user input"""
    return st.sidebar.text_input("Enter Your API Key:", type="password")

# Get API Key
api_key = get_api_key()
if api_key:
    st.session_state.api_key = api_key

# Database Selection
db_options = ["merchant_data_dubai.db", "merchant_data_singapore.db"]
new_selected_db = st.sidebar.selectbox("Select Database:", db_options, index=db_options.index(st.session_state.selected_db))

# Check if the database selection has changed
if new_selected_db != st.session_state.selected_db:
    st.session_state.selected_db = new_selected_db
    st.session_state.db_initialized = False  # Reset database initialization
    st.sidebar.success(f"✅ Switched to database: {st.session_state.selected_db}")

# Model Selection
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])

# Email Template Selection
template_options = ["email_task_description1.txt", "email_task_description2.txt", "email_task_description3.txt"]
st.session_state.selected_template = st.sidebar.selectbox("Select Email Template:", template_options, index=template_options.index(st.session_state.selected_template))
st.sidebar.success(f"✅ Selected Template: {st.session_state.selected_template}")

# Initialize SQL Database and Agent
if st.session_state.selected_db and api_key and not st.session_state.db_initialized:
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0,
            model_name=model_name,
            api_key=st.session_state.api_key
        )

        # Initialize SQLDatabase
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{st.session_state.selected_db}", sample_rows_in_table_info=3)

        # Create SQL Agent
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            db=st.session_state.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        st.session_state.db_initialized = True  # Mark database as initialized
        st.sidebar.success("✅ Database and LLM Connected Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Function to render the "Enter Query" section
def render_query_section():
    st.markdown("#### Ask questions about your database:", unsafe_allow_html=True)
    
    # Predefined questions
    predefined_questions = [
        "Give first five merchant names and their emails",
        "Give first 10 merchants, their emails and their image urls",
        "Who are You?"
    ]
    
    # Display buttons for predefined questions
    st.markdown("**Predefined Questions:**")
    for question in predefined_questions:
        if st.button(question, key=f"predefined_{question}"):
            st.session_state.user_query = question  # Store the question in session state
            st.session_state.trigger_rerun = True  # Trigger a re-run to process the query
    
    # Text area for user input
    user_query = st.text_area("Enter your query:", placeholder="E.g., Show top 10 merchants and their emails.", key=f"query_{len(st.session_state.interaction_history)}", value=st.session_state.get('user_query', ''))
    
    if st.button("Run Query", key=f"run_query_{len(st.session_state.interaction_history)}"):
        if user_query:
            with st.spinner("Running query..."):
                try:
                    # Define company details and agent role
                    company_details = """
                    Pulse iD is a fintech company specializing in merchant solutions and personalized marketing. 
                    As a marketing agent for Pulse iD, my role is to assist you in querying the merchant database 
                    and generating personalized emails for marketing purposes.
                    """

                    # Prepend company details to the user's query
                    full_query = f"{company_details}\n\nUser Query: {user_query}"

                    # Execute the query using the agent
                    result = st.session_state.agent_executor.invoke(full_query)
                    st.session_state.raw_output = result['output'] if isinstance(result, dict) else result
                    
                    # Process raw output using an extraction agent 
                    extractor_llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.session_state.api_key)
                    extractor_agent = Agent(
                        role="Data Extractor",
                        goal="Extract merchants, emails from the raw output if they are only available.",
                        backstory="You are an expert in extracting structured information from text.",
                        provider="Groq",
                        llm=extractor_llm 
                    )
                    
                    extract_task = Task(
                        description=f"Extract a list of 'merchants' and their 'emails', 'image urls' from the following text:\n\n{st.session_state.raw_output}",
                        agent=extractor_agent,
                        expected_output="A structured list of merchants, their associated email addresses, image URLs (except 'Pulse id') extracted from the given text. plrease If any of merchants, emails are unavailable, return 'errorhappened'."
                    )
                    
                    # Crew execution for extraction 
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()
                    st.session_state.extraction_results = extraction_results if extraction_results else ""
                    st.session_state.merchant_data = st.session_state.extraction_results
                    
                    # Append the query and results to the interaction history
                    st.session_state.interaction_history.append({
                        "type": "query",
                        "content": {
                            "query": user_query,
                            "raw_output": st.session_state.raw_output,
                            "extraction_results": st.session_state.extraction_results
                        }
                    })
                    
                    # Trigger a re-run to update the UI
                    st.session_state.trigger_rerun = True
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")

# Display Interaction History
if st.session_state.interaction_history:
    st.markdown("### Interaction History:", unsafe_allow_html=True)
    for idx, interaction in enumerate(st.session_state.interaction_history):
        if interaction["type"] == "query":
            st.markdown(f"#### Query: {interaction['content']['query']}")
            st.markdown("**Raw Output:**")
            st.write(interaction['content']['raw_output'])
            
            # Only display extracted merchants if there is data and it does not contain ''
            if interaction['content']['extraction_results'] and interaction['content']['extraction_results'].raw and 'errorhappened' not in interaction['content']['extraction_results'].raw:
                st.markdown("**Extracted Merchants:**")
                st.write(interaction['content']['extraction_results'].raw)
                
                # Show the "Generate Emails" button for this specific interaction
                if st.button(f"Generate Emails For Above Extracted Merchants", key=f"generate_emails_{idx}"):
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

                            # Read the task description from the selected template file
                            description_file_path = f"email_descriptions/{st.session_state.selected_template}"
                            email_task_description = read_email_task_description(description_file_path)

                            # Email generation task using extracted results 
                            task = Task(
                                description=email_task_description.format(merchant_data=interaction['content']['extraction_results'].raw),
                                agent=email_agent,
                                expected_output="Marketing emails for each selected merchant, tailored to their business details. Please tell the user to extract merchants first if there are no data available for emails, merchant names."
                            )

                            # Crew execution 
                            crew = Crew(agents=[email_agent], tasks=[task], process=Process.sequential)
                            email_results = crew.kickoff()
                            
                            # Display results 
                            if email_results.raw:
                                email_body = email_results.raw  # Get the raw email content
                                
                                # Function to extract image URL from email body
                                def extract_image_url(email_body):
                                    url_pattern = r'https?://[^\s]+'
                                    urls = re.findall(url_pattern, email_body)
                                    return urls[0] if urls else None

                                # Extract image URL from the email body
                                image_url = extract_image_url(email_body)

                                # Insert image into the email body at a specific position (after "Dear Merchant Name")
                                if image_url:
                                    modified_email_body = email_body.replace("Dear", f"Dear,<br><img src='{image_url}' style='max-width: 100%;' />")
                                    email_body = modified_email_body
                                
                                # Append the generated email to the interaction history
                                st.session_state.interaction_history.append({
                                    "type": "email",
                                    "content": email_body
                                })
                                
                                # Trigger a re-run to update the UI
                                st.session_state.trigger_rerun = True

                        except Exception as e:
                            st.error(f"Error generating emails: {str(e)}")
        
        elif interaction["type"] == "email":
            st.markdown("#### Generated Email:")
            st.markdown(interaction['content'], unsafe_allow_html=True)
        
        st.markdown("---")

# Always render the "Ask questions about your database" section
render_query_section()

# Trigger a re-run if needed
if st.session_state.trigger_rerun:
    st.session_state.trigger_rerun = False  # Reset the trigger
    st.rerun()  # Force a re-run of the script

# Footer Section 
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Powered by <strong>Pulse iD</strong> | Built with 🐍 Python and Streamlit</div>",
    unsafe_allow_html=True 
)
