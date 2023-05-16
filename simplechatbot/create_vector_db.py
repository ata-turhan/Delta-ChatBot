from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import os
import sys
from dotenv import load_dotenv, find_dotenv
from chromadb.utils import embedding_functions


def main():
    load_dotenv(find_dotenv())

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    persist_directory = "Vector-DB"

    embeddings = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
    )

    vector_db = Chroma.from_documents(
        documents=texts,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_db.persist()


if __name__ == "__main__":
    main()
