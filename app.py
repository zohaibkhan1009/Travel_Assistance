import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel
from datetime import datetime

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_llm():
    return LLM(model="gpt-3.5-turbo", api_key=openai.api_key)

# Define input schema for the web search tool
class SearchQuery(BaseModel):
    query: str

# Web search tool with explicit args_schema
search_tool = StructuredTool(
    name="Web Search",
    description="Fetches web search information based on a query.",
    func=DuckDuckGoSearchResults(num_results=5).run,
    args_schema=SearchQuery,
)

# Define agents
guide_expert = Agent(
    role="City Local Guide Expert",
    goal="Provides information on things to do in the city based on user's interests.",
    backstory="A local expert passionate about sharing hidden gems.",
    tools=[search_tool],
    verbose=True,
    max_iter=5,
    llm=get_llm(),
    allow_delegation=False,
)

location_expert = Agent(
    role="Travel Trip Expert",
    goal="Gathers helpful information about {destination_city} and travel details.",
    backstory="A seasoned traveler who knows logistics and attractions.",
    tools=[search_tool],
    verbose=True,
    max_iter=5,
    llm=get_llm(),
    allow_delegation=False,
)

planner_expert = Agent(
    role="Travel Planning Expert",
    goal="Compiles all gathered information into a detailed travel plan.",
    backstory="An organizational wizard who creates seamless itineraries.",
    tools=[search_tool],
    verbose=True,
    max_iter=5,
    llm=get_llm(),
    allow_delegation=False,
)

# User input
def get_user_inputs():
    from_city = st.text_input("Traveling From:", "India")
    destination_city = st.text_input("Destination City:", "Rome")
    date_from = st.date_input("Arrival Date")
    date_to = st.date_input("Departure Date")
    interests = st.text_input("Interests:", "Sightseeing and good food")
    return from_city, destination_city, date_from, date_to, interests

# Define tasks
def create_tasks(from_city, destination_city, date_from, date_to, interests):
    num_days = (date_to - date_from).days + 1  # Ensure single-day trips are accounted for
    
    location_task = Task(
        description=f"""
        Collect comprehensive data on accommodations, cost of living, travel advisories, 
        weather conditions, and local events in {destination_city}.
        """,
        expected_output="Detailed markdown report with travel logistics.",
        agent=location_expert,
        output_file='city_report.md',
    )
    
    guide_task = Task(
        description=f"""
        Generate a personalized city guide for {destination_city} covering attractions, restaurants, entertainment, 
        and cultural spots based on {interests}.
        """,
        expected_output="Markdown itinerary with places of interest.",
        agent=guide_expert,
        output_file='guide_report.md',
    )
    
    planner_task = Task(
        description=f"""
        Compile all collected data into a structured travel plan for {destination_city} spanning {num_days} days.
        """,
        expected_output="Detailed markdown itinerary with daily schedules.",
        context=[location_task, guide_task],
        agent=planner_expert,
        output_file='travel_plan.md',
    )
    
    return [location_task, guide_task, planner_task]

# Run Crew
def run_crew(tasks):
    crew = Crew(
        agents=[location_expert, guide_expert, planner_expert],
        tasks=tasks,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
        verbose=True,
    )
    return crew.kickoff()

# Streamlit UI
st.title("Smart Travel Assistant")
from_city, destination_city, date_from, date_to, interests = get_user_inputs()

tasks = create_tasks(from_city, destination_city, date_from, date_to, interests)
if st.button("Generate Travel Plan"):
    result = run_crew(tasks)
    st.success("Travel plan generated!")
    st.markdown(result)

if st.button("Clear Chat History"):
    st.experimental_rerun()

if st.button("Terminate API Usage"):
    os.environ.pop("OPENAI_API_KEY", None)
    st.warning("API Key removed. Restart required.")
