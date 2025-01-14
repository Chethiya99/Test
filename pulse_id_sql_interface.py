# Function to render the "Enter Query" section with predefined questions
def render_query_section():
    st.markdown("#### Ask questions about your database:", unsafe_allow_html=True)
    
    # Predefined questions
    predefined_questions = [
        "Give first five merchant names and their emails",
        "Give first 10 merchants, their emails, and their image URLs",
        "Who are You?",
        "Show top 10 merchants by transaction volume",
        "List merchants with email addresses containing 'gmail.com'",
        "Find merchants located in Dubai",
        "Retrieve merchants with the highest number of transactions",
        "Get merchants with missing email addresses",
        "List merchants with the oldest establishment dates",
        "Find merchants with the most recent transactions"
    ]
    
    # Display buttons for predefined questions
    st.markdown("**Predefined Questions:**")
    cols = st.columns(3)  # Adjust the number of columns as needed
    for idx, question in enumerate(predefined_questions):
        with cols[idx % 3]:  # Distribute buttons across columns
            if st.button(question, key=f"predefined_question_{idx}"):
                st.session_state.current_query = question  # Store the selected question in session state
    
    # Text area for user input or predefined question
    user_query = st.text_area(
        "Enter your query:",
        value=st.session_state.get("current_query", ""),  # Populate with the selected question
        placeholder="E.g., Show top 10 merchants and their emails.",
        key=f"query_{len(st.session_state.interaction_history)}"
    )
    
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
                        expected_output="A structured list of merchants, their associated email addresses, image URLs (except 'Pulse id') extracted from the given text. If any of merchants and emails are unavailable, return 'errorhappened'."
                    )
                    
                    # Crew execution for extraction 
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()
                    
                    # Debug: Print extraction results to console
                    print("Extraction Results:", extraction_results)
                    
                    # Store extraction results in session state
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
            
            # Check if extraction results exist and are valid
            if interaction['content']['extraction_results'] and hasattr(interaction['content']['extraction_results'], 'raw'):
                extraction_results_raw = interaction['content']['extraction_results'].raw
                
                # Debug: Print extraction results to console
                print("Extraction Results Raw:", extraction_results_raw)
                
                # Display extracted merchants if data is available
                if extraction_results_raw and extraction_results_raw != "errorhappened":
                    st.markdown("**Extracted Merchants:**")
                    st.write(extraction_results_raw)
                    
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
                                    description=email_task_description.format(merchant_data=extraction_results_raw),
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
