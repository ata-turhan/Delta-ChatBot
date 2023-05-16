import sys
import os

from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.chains.question_answering import load_qa_chain
from chromadb.utils import embedding_functions
from chromadb.config import Settings


def main():
    load_dotenv(find_dotenv())
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    FULL_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(FULL_PATH, "Vector-DB")

    embeddings = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
    )

    loader = TextLoader(
        "C:/Users/NEO/Desktop/Projects/SimpleChatbot/input/tel.txt", encoding="UTF8"
    )
    documents = loader.load()
    char_text_splitter = MarkdownTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=3000,
        chunk_overlap=500,
    )
    texts = char_text_splitter.split_documents(documents)

    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",  # we'll store as parquet files/DuckDB
        persist_directory=DB_DIR,  # location to store
        anonymized_telemetry=False,  # optional but showing how to toggle telemetry
    )

    vector_store = None

    if not os.path.exists(DB_DIR):
        vector_store = Chroma.from_documents(
            texts,
            # embeddings,
            persist_directory=DB_DIR,
            client_settings=client_settings,
            collection_name="Store",
        )
        vector_store.persist()
    else:
        vector_store = Chroma(
            collection_name="Store",
            persist_directory=DB_DIR,
            # embedding_function=embeddings,
            client_settings=client_settings,
        )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # chain = load_qa_chain(llm, chain_type="stuff")
    # result = chain.run(input_documents=docs, question=query)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = "What are the SMS guidelines for Ireland?"
    print(qa.run(query))
    query = "What country uses the country code 52?"
    print(qa.run(query))
    query = "What is the MCC for Serbia?"
    print(qa.run(query))
    query = "Is alphanumeric SMS supported in China?"
    print(qa.run(query))
    query = "Are there any SMS restrictions in Cambodia?"
    print(qa.run(query))
    query = "Is alphanumeric registration required in Poland?"
    print(qa.run(query))
    query = "Are you okay?"
    print(qa.run(query))
    query = "What country uses the country code 90?"
    print(qa.run(query))

    # vectorstore=vector_db,


if __name__ == "__main__":
    main()
