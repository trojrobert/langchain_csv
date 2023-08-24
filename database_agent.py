from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SQLDatabaseSequentialChain
import streamlit as st

def connect_to_db(username,
                  password,
                  host, 
                  port):

    username=str(username)
    password=str(password)
    host=str(host)
    port=int(port)

    print(username)
    print(password)
    print(host)
    print(port)
    
    mydatabase = "postgres"
    pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{int(port)}/{mydatabase}"
    db = SQLDatabase.from_uri(pg_uri)

    # llm = OpenAI(temperature=0, openai_api_key=st.session_state.OPENAI_API_KEY, model_name='gpt-3.5-turbo')
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # db_chain = SQLDatabaseSequentialChain(llm=llm, database=db, verbose=True, top_k=3)
    db_chain = SQLDatabaseChain(llm=llm,database=db,verbose=True)

    return db_chain
