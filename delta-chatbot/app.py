import os
import tempfile

import streamlit as st
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreToolkit,
    create_vectorstore_agent,
)
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from PIL import Image

# Storing the chat
if "user" not in st.session_state:
    st.session_state.user = []
if "bot" not in st.session_state:
    st.session_state.bot = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None


def start_chat():
    # Set the title and subtitle of the app
    st.title("🦜🔗 PDF-Chat: Interact with Your PDFs in a Conversational Way")
    st.subheader(
        "Load your PDF, ask questions, and receive answers directly from the document."
    )

    # Load the image
    image = Image.open("PDF-Chat App.png")
    st.image(image)

    # Loading the Pdf file and return a temporary path for it
    st.subheader("Upload your pdf")
    uploaded_file = st.file_uploader(
        "", type=(["pdf", "tsv", "csv", "txt", "tab", "xlsx", "xls"])
    )

    temp_file_path = os.getcwd()
    while uploaded_file is None:
        return

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        st.write("Full path of the uploaded file:", temp_file_path)

    # Create instance of OpenAI LLM
    llm = OpenAI(temperature=0.1, verbose=True)
    embeddings = OpenAIEmbeddings()

    # Create and load PDF Loader
    loader = PyPDFLoader(temp_file_path)
    # Split pages from pdf
    pages = loader.load_and_split()

    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(pages, embeddings, collection_name="Pdf")

    # Create vectorstore info object
    vectorstore_info = VectorStoreInfo(
        name="Pdf",
        description=" A pdf file to answer your questions",
        vectorstore=store,
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm, toolkit=toolkit, verbose=True
    )

    if prompt := st.text_input("Input your prompt here"):
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt)
            # Write out the first
            st.write(search[0][0].page_content)


def is_api_key_valid(openai_api_key: str):
    if openai_api_key is None or not openai_api_key.startswith("sk-"):
        st.warning("Lütfen geçerli bir OpenAI API Key'i girin!", icon="⚠")
        return False
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True


def main():
    st.set_page_config(
        page_title="🤖 Delta ChatBot",
        page_icon="🤖",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/olympian-21",
            "Report a bug": None,
            "About": "This is a chat bot to query documents",
        },
    )

    st.markdown(
        "<center><h1>Delta - Document ChatBot</h1></center> <br> <br>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<center><h3>Confiugarations</h3></center> <br> <br>",
        unsafe_allow_html=True,
    )

    st.sidebar.text_input("Please enter the OpenAI API Key:", key="openai_api")

    if is_api_key_valid(st.session_state.openai_api):
        st.sidebar.success("OpenAI API Key was received successfully.")
        start_chat()


if __name__ == "__main__":
    main()
