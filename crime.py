import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sqlalchemy_text
from dotenv import load_dotenv
from langchain.globals import set_verbose
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load environment variables
load_dotenv()

# Set up OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Database setup
engine = create_engine("sqlite:///crime.db")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")

# Enable verbose logging
set_verbose(True)

# Create SQL agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools")

# Data dictionary for context
data_dictionary = """
| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Report ID                | Unique identifier for each crime report                                     |
| Report Datetime          | Date and time when the crime was reported                                   |
| Reporting Station        | Name of the police station handling the case                                |
| Classification           | Classification of the crime (e.g., Known, Unknown)                          |
| Incident Latitude        | Geographic latitude of the incident location                                |
| Incident Longitude       | Geographic longitude of the incident location                               |
| Incident Datetime        | Date and time when the crime incident occurred                              |
| Charge Type              | Type of charge associated with the crime                                    |
| Accused Name             | Name of the accused individual                                              |
| Accused Passport         | Passport number of the accused individual                                   |
| Accused ID               | Unique identification number of the accused (if available)                  |
| Accused Age              | Age of the accused individual                                               |
| Accused Marital Status   | Marital status of the accused individual                                    |
| Accused Nationality      | Nationality of the accused individual                                       |
| Accused Status           | Residency status of the accused (e.g., Resident, Visitor)                   |
| Report Status            | Current status of the crime report (e.g., Under Investigation)              |
| Accused Custody          | Indicator of whether the accused is in custody (1 = In Custody, 0 = Not in Custody) |
"""

# Streamlit UI setup
st.title("Crime Analytics Chatbot")
st.write("Ask questions about crime and explore analytics.")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "follow_up_prompts" not in st.session_state:
    st.session_state.follow_up_prompts = []
if "follow_up_triggered" not in st.session_state:
    st.session_state.follow_up_triggered = False
if "input_field" not in st.session_state:
    st.session_state["input_field"] = ""

# Function to handle user input
def handle_user_input(user_input):
    if user_input:
        # Add the data dictionary to the input for better context
        input_text = f"Refer to the following data dictionary for context:\n\n{data_dictionary}\n\n{user_input}"

        # Query the RAG model for a natural language answer first
        response = agent_executor.invoke({"input": input_text})
        result = response["output"]

        # Check conditions to run queries
        # 1. If user asked about "top"/"highest" stations
        if "top" in user_input or "highest" in user_input:
            query = sqlalchemy_text("""
            SELECT reporting_station, COUNT(report_id) AS Crime_Count
            FROM crime
            GROUP BY reporting_station
            ORDER BY Crime_Count DESC
            LIMIT 10
            """)
            with engine.connect() as connection:
                data = pd.read_sql(query, connection)

            # Append station info to the result
            station_info = []
            for idx, row in data.iterrows():
                station_info.append(f"{row['reporting_station']} - {row['Crime_Count']} reports")
            result += "\n\n" + "\n".join(station_info)

            # Plot a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(data["reporting_station"], data["Crime_Count"])
            ax.set_xlabel("Police Station", color="white")
            ax.set_ylabel("Crime Count", color="white")
            ax.set_title("Top 10 Police Stations by Crime Count", color="white")
            plt.xticks(rotation=45)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            fig.patch.set_facecolor('#2B2B2B')
            ax.set_facecolor('#2B2B2B')
            st.pyplot(fig)

        # 2. If user wants to "compare" or mentions "all police stations"
        elif "compare" in user_input or "all police stations" in user_input:
            query = sqlalchemy_text("""
            SELECT reporting_station, COUNT(report_id) AS Crime_Count
            FROM crime
            GROUP BY reporting_station
            ORDER BY Crime_Count DESC
            """)
            with engine.connect() as connection:
                data = pd.read_sql(query, connection)

            # Append station info with counts to the result
            station_info = []
            for idx, row in data.iterrows():
                station_info.append(f"{row['reporting_station']} - {row['Crime_Count']} reports")
            result += "\n\n" + "\n".join(station_info)

        # After queries and modifications, store final result in conversation
        st.session_state.conversation.append(("Police Bot", result))

        # Generate Follow-Up Prompts
        st.session_state.follow_up_prompts = []
        if "Barsha" in user_input or "police station" in user_input:
            st.session_state.follow_up_prompts = [
                "Would you like to compare the crimes between this year and last year?",
                "Would you like to compare all police stations?",
                "Would you like to see a breakdown of crime types in Barsha?",
                "Would you like to visualize total crime in each station?"
            ]
        elif "top" in user_input or "highest" in user_input:
            st.session_state.follow_up_prompts = [
                "Would you like to see trends for the top police stations?",
                "Would you like to analyze why these stations have high crime rates?",
                "Would you like to compare crime numbers over time?",
                "Would you like a breakdown of crime categories in these stations?"
            ]

# Handle follow-up prompt first
if st.session_state.follow_up_triggered:
    st.session_state["input_field"] = st.session_state.follow_up_triggered
    st.session_state.follow_up_triggered = False
    user_input = st.session_state["input_field"]
    st.session_state["input_field"] = ""  # Clear the input after usage
    handle_user_input(user_input)

# If there's user input in input_field, handle it before displaying conversation
if st.session_state["input_field"]:
    user_input = st.session_state["input_field"]
    st.session_state["input_field"] = ""  # Clear input after usage
    handle_user_input(user_input)

# Now display conversation and follow-up prompts after processing input
for speaker, text in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Police Bot:** {text}")

if st.session_state.follow_up_prompts:
    st.write("**Follow-Up Suggestions:**")
    columns = st.columns(len(st.session_state.follow_up_prompts))
    for idx, prompt in enumerate(st.session_state.follow_up_prompts):
        if columns[idx].button(prompt):
            st.session_state.follow_up_triggered = prompt

# Finally, display the input field
st.text_input("Type your message and press Enter", key="input_field")
