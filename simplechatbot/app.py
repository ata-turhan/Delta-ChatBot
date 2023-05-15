import sys
import os
import openai
import pinecone

import langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

import magic
import nltk
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

nltk.download("punkt")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def main():
    # print(f"Command line argument - '{sys.argv[1]}' is received succesfully.")
    loader = TextLoader(
        "C:/Users/NEO/Desktop/Projects/SimpleChatbot/input/tel.txt", encoding="UTF8"
    )

    index = VectorstoreIndexCreator().from_loaders([loader])
    query = "What are the SMS guidelines for Ireland?"
    print(index.query(query))
    query = "What country uses the country code 52?"
    print(index.query(query))
    query = "What is the MCC for Serbia?"
    print(index.query(query))
    query = "Is alphanumeric SMS supported in China?"
    print(index.query(query))
    query = "Are there any SMS restrictions in Cambodia?"
    print(index.query(query))
    query = "Is alphanumeric registration required in Poland?"
    print(index.query(query))


if __name__ == "__main__":
    main()
