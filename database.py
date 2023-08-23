from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
import streamlit as st

def connect_to_db():


    # pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
    # db = SQLDatabase.from_uri(pg_uri)

    # # llm = OpenAI(temperature=0, openai_api_key=st.session_state.OPENAI_API_KEY, model_name='gpt-3.5-turbo')
    # llm = OpenAI(temperature=0)
    # # db_chain = SQLDatabaseSequentialChain(llm=llm, database=db, verbose=True, top_k=3)
    # db_chain = SQLDatabaseChain(llm=llm,database=db,verbose=True)

    # question_1 = db_chain.run("How many employees in employees tables")
    # print(question_1)

    # question = "list all the tables in the column?" 
    # # use db_chain.run(question) instead if you don't have a prompt
    # db_chain.run(question)

    # print(db_chain)
    pass