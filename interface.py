import streamlit as st
import pandas as pd
import json
import os 

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from pandas_agent import query_agent, create_agent
from database_agent import connect_to_db
from utils import store_data_as_vectors
from langchain.document_loaders.csv_loader import CSVLoader


# def decode_response(response: str) -> dict:
#     """This function converts the string response from the model to a dictionary object.

#     Args:
#         response (str): response from the model

#     Returns:
#         dict: dictionary with response data
#     """
#     return json.loads(response)


# def write_response(response_dict: dict):
#     """
#     Write a response from an agent to a Streamlit app.

#     Args:
#         response_dict: The response from the agent.

#     Returns:
#         None.
#     """

#     # Check if the response is an answer.
#     if "answer" in response_dict:
#         st.write(response_dict["answer"])

#     # Check if the response is a bar chart.
#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         df = pd.DataFrame(data)
#         df.set_index("columns", inplace=True)
#         st.bar_chart(df)

#     # Check if the response is a line chart.
#     if "line" in response_dict:
#         data = response_dict["line"]
#         df = pd.DataFrame(data)
#         df.set_index("columns", inplace=True)
#         st.line_chart(df)

#     # Check if the response is a table.
#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)


# sidebar interface to get OpenAI API key
with st.sidebar:
    st.text("Enter your OpenAI API Key")
    st.session_state.OPENAI_API_KEY = st.text_input(label='*We do NOT store and cannot view your API key*',
                                                    placeholder='',
                                                    type="password",
                                                    help='You can find your Secret API key at \
                                                            https://platform.openai.com/account/api-keys')
    

# Title and banner for the home page    
st.title(" 👱‍♀️ ZOE")

file_uploaded = None 
loader = None

# Create option to select the type of document to upload
document_type = st.selectbox(
    "Select upload type",
    ("",
    ".csv",
    ".pdf",
    ".txt",
    "database")
)


if document_type:
    if document_type == ".csv":
        file_uploaded = st.file_uploader("Upload .csv file", type="csv")
        if file_uploaded: 
            loader = st.selectbox(
                "Select loader",
                ("",
                "csv loader",
                "pandas loader")
                )
            
    if document_type == ".pdf":
        file_uploaded = st.file_uploader("Upload .pdf file", type="pdf")
        st.write(".pdf file uploaded")
        st.write("Still on development")
        st.stop()

    if document_type == ".txt":
        file_uploaded = st.file_uploader("Upload .txt file", type="txt")
        st.write(".txt file uploaded")
        st.write("Still on development")
        st.stop()

    if document_type == "database":
        st.write("database selected")
        st.warning("This only work with postgres database")
        
        st.session_state.username = st.text_input(label="username")
        st.session_state.password = st.text_input(label="password")
        st.session_state.host = st.text_input(label="host")
        st.session_state.port = st.text_input(label="port") 
        file_uploaded = "database"

       
    
# else:
#     st.write("Please select a document type")


if file_uploaded==".csv" and loader: 
    query = st.text_area("Insert your query")

    if st.button("Submit Query", type="primary"):
        if not st.session_state.OPENAI_API_KEY:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY
        # Create an OpenAI object.
        llm = OpenAI(openai_api_key=st.session_state.OPENAI_API_KEY)

        if document_type==".csv":
            # Load data csv file with Pandas
            dataframe = pd.read_csv(file_uploaded)

            if loader=="csv loader":

                # Convert pandas to csv and load with it csv loader
                dataframe.to_csv("data.csv")
                loadfile = CSVLoader("data.csv")
                csv_data = loadfile.load()

                vectordb = store_data_as_vectors(csv_data)

                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=vectordb.as_retriever()
                    )

                response = qa_chain({"query": query})

                st.write(response['result'])

            if loader=="pandas loader":

                # Create an agent from the CSV file.
                agent = create_agent(dataframe, llm)

                # Query the agent.
                response = query_agent(agent=agent, query=query)

                st.write(response)

if file_uploaded=="database":

    query = st.text_area("Insert your query")

    if st.button("Submit Query", type="primary"):
        if not st.session_state.OPENAI_API_KEY:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    
        os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY

        if not st.session_state.username:
            st.info("Please add database username")
        
        if not st.session_state.password:
            st.info("Please add database password")

        if not st.session_state.port:
            st.info("Please add database port")
        
        if not st.session_state.host:
            st.info("Please add database host")

        # Create Database agent to connect to database 
        db_connection = connect_to_db(username=st.session_state.username,
                    password=st.session_state.password,
                    host=st.session_state.host,
                    port=st.session_state.port) 
        
        response = db_connection.run(query)
        st.write(response)

# st.write("Please upload your CSV file below.")

# data = st.file_uploader("Upload a CSV")

# query = st.text_area("Insert your query")

# if st.button("Submit Query", type="primary"):
#     if not st.session_state.OPENAI_API_KEY:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()

#     os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY

#     # Database agent test 
#     connect_to_db()

#     # Load data csv file with Pandas
#     dataframe = pd.read_csv(data)

#     # Create an agent from the CSV file.
#     agent = create_agent(data)
    
#     # Test csv loader
#     dataframe.to_csv("data.csv")
#     loadfile = CSVLoader("data.csv")
#     data = loadfile.load()

#     # Query the agent.
#     response = query_agent(agent=agent, query=query)

#     # Decode the response.
#     decoded_response = decode_response(response)

#     # Write the response to the Streamlit app.
#     # write_response(decoded_response)
#     st.write(response)