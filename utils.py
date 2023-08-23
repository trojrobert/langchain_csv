
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings

def create_chunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, 
                                                chunk_overlap=20, 
                                                length_function=len)
    
    chunks = text_splitter.create_documents(documents)
    return chunks

def store_data_as_vectors(documents):

    # Convert the chunks to embeddings
    embeddings = OpenAIEmbeddings()

    # vectorstore = FAISS.from_documents(documents, embeddings)


    collection_name = "pdf_collection"
    store_path = "vector_store/chroma"
    vectordb = Chroma.from_documents(
        documents=documents,
        persist_directory=store_path, 
        embedding=embeddings)
    vectordb.persist()

    return vectordb
