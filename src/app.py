import os
import sys

from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma


def main():
    load_dotenv(find_dotenv())
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    FULL_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(FULL_PATH, "Vector-DB")
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )
    vector_store = None

    file_path = "input/telnyx-doc.json"
    main_dir = os.path.dirname(FULL_PATH)
    FILE_DIR = os.path.join(main_dir, file_path)
    loader = TextLoader(FILE_DIR, encoding="utf-8")
    documents = loader.load()
    char_text_splitter = MarkdownTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
    )
    texts = char_text_splitter.split_documents(documents)

    if not os.path.exists(DB_DIR):
        vector_store = Chroma.from_documents(
            texts,
            persist_directory=DB_DIR,
            client_settings=client_settings,
            collection_name="Store",
        )
        vector_store.persist()
    else:
        vector_store = Chroma(
            collection_name="Store",
            persist_directory=DB_DIR,
            client_settings=client_settings,
        )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    prompt_template = """
    <|SYSTEM|>#
    - You are a helpful, polite, fact-based agent for answering questions about Telnyx SMS Guidelines Documentation.
    <|USER|>
    Please answer the following question using the context provided. If you don't know the answer, just say that "The answer to this question cannot be found in the document". Base your answer on the context below. Say "The answer to this question cannot be found in the document" if the answer does not appear to be in the context below.

    QUESTION: {question}
    CONTEXT:
    {context}

    ANSWER: <|ASSISTANT|>
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )

    try:
        query = sys.argv[1]
        print(qa.run(query))
    except Exception as e:
        print("Please provide a valid query.")
        print(e)


if __name__ == "__main__":
    main()
