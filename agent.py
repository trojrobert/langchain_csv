from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
import streamlit as st



def create_agent(dataframe: pd.DataFrame, llm):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """

    # # Create an OpenAI object.
    # llm = OpenAI(openai_api_key=st.session_state.OPENAI_API_KEY)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, dataframe, verbose=False)


def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    # prompt = (
    #     """
    #         For the following query, if it requires drawing a table, reply as follows:
    #         {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

    #         If the query requires creating a bar chart, reply as follows:
    #         {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            
    #         If the query requires creating a line chart, reply as follows:
    #         {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            
    #         There can only be two types of chart, "bar" and "line".
            
    #         If it is just asking a question that requires neither, reply as follows:
    #         {"answer": "answer"}
    #         Example:
    #         {"answer": "The title with the highest rating is 'Gilead'"}
            
    #         If you do not know the answer, reply as follows:
    #         {"answer": "I do not know."}
            
    #         Return all output as a string.
            
    #         All strings in "columns" list and data list, should be in double quotes,
            
    #         For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}
            
    #         Lets think step by step.
            
    #         Below is the query.
    #         Query: 
    #         """
    #     + query
    # )


    # Run the prompt through the agent.
    response = agent.run(query)

    # Convert the response to a string.
    # return response.__str__()
    return response

