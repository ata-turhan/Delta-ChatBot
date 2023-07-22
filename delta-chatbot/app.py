import os
import tempfile
import time
from collections import OrderedDict

import openai
import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from modules.utils import add_bg_from_local, set_page_config

STREAMING_INTERVAL = 0.01


if "messages" not in st.session_state:
    st.session_state.messages = OrderedDict()


def is_api_key_valid(model_host: str, api_key: str):
    if api_key is None:
        st.sidebar.warning("Lütfen geçerli bir API keyi girin!", icon="⚠")
        return False
    elif model_host == "openai" and not api_key.startswith("sk-"):
        st.sidebar.warning(
            "Lütfen geçerli bir OpenAI API keyi girin!", icon="⚠"
        )
        return False
    elif model_host == "huggingface" and not api_key.startswith("hf_"):
        st.sidebar.warning(
            "Lütfen geçerli bir HuggingFace API keyi girin!", icon="⚠"
        )
        return False
    else:
        if model_host == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
        else:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        return True


def create_vector_store_retriever(model_host, chunked_documents):
    embeddings = (
        OpenAIEmbeddings()
        if model_host == "openai"
        else HuggingFaceHubEmbeddings()
    )
    vector_store = Chroma.from_documents(chunked_documents, embeddings)
    return vector_store.as_retriever()


def create_llm(model):
    return (
        ChatOpenAI(
            model_name=model.split("/")[1],
            temperature=0,
        )
        if model.startswith("openai")
        else HuggingFaceHub(
            repo_id=model,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 4096,
            },
        )
    )


def create_main_prompt():
    return """
    <|SYSTEM|>#
    You are a bot that lets your user talk to the documents he has uploaded.
    - Answer using the information in the documents provided.
    - If the answers are not in the documentation, you should say "Sorry, I could not find the answer to this question
    in your uploaded files.".
    <|USER|>
    Now the user asks you a question. Using your knowledge of the context and chat history provided to you, answer this question
    clearly and precisely.

    QUESTION: {question}
    CONTEXT:
    {context}

    ANSWER: <|ASSISTANT|>
    """


def create_retrieval_qa(llm, prompt_template, retriever):
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    combine_docs_chain_kwargs = {"prompt": PROMPT}
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        memory=memory,
    )


def prepare_documents(uploaded_files):
    temp_files = []
    temp_dir = tempfile.TemporaryDirectory()
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_files.append(temp_file_path)

    documents = []
    for temp_file in temp_files:
        if temp_file.endswith(".pdf"):
            loader = PyPDFLoader(temp_file)
        elif temp_file.endswith(".doc") or temp_file.endswith(".docx"):
            loader = Docx2txtLoader(temp_file)
        elif temp_file.endswith(".txt"):
            loader = TextLoader(temp_file)
        documents.extend(loader.load())

    character_splitter = CharacterTextSplitter(
        chunk_size=512, chunk_overlap=32
    )
    return character_splitter.split_documents(documents)


def main():
    set_page_config()

    background_img_path = os.path.join("static", "background", "Bot.png")
    sidebar_background_img_path = os.path.join(
        "static", "background", "Lila Gradient.png"
    )
    page_markdown = add_bg_from_local(
        background_img_path=background_img_path,
        sidebar_background_img_path=sidebar_background_img_path,
    )
    st.markdown(page_markdown, unsafe_allow_html=True)

    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 60px;'>
         📝 Delta - Document ChatBot
         </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 20px;'>
        You can upload your files and start to chat with them
        </h1>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<center><h1>Sohbet Botu Ayarları</h1></center> <br>",
        unsafe_allow_html=True,
    )

    model = st.sidebar.selectbox(
        "Lütfen bir LLM seçin",
        [
            "<Seçiniz>",
            "openai/gpt-3.5-turbo",
            "google/flan-t5-xxl",
            "databricks/dolly-v2-3b",
            "Writer/camel-5b-hf",
            "Salesforce/xgen-7b-8k-base",
            "tiiuae/falcon-40b",
            "bigscience/bloom",
        ],
    )
    if model == "<Seçiniz>":
        st.sidebar.warning("Lütfen bir model seçin.")
        _, center_war_col, _ = st.columns([2, 5, 1])
        center_war_col.warning(
            "Lütfen sol taraftaki panelden bot için gerekli ayarlamaları yapın."
        )
        return
    else:
        api_key = st.sidebar.text_input(
            f"Lütfen {model} API keyini girin",
        )
        model_host = "openai" if model.startswith("openai") else "huggingface"
        if is_api_key_valid(model_host, api_key):
            st.sidebar.success("API keyi başarıyla alındı.")
        else:
            _, center_war_col, _ = st.columns([2, 5, 1])
            center_war_col.warning(
                "Lütfen sol taraftaki panelden bot için gerekli ayarlamaları yapın."
            )
            return

    uploaded_files = st.file_uploader(
        "You can upload any number of documents you want to chat with",
        type=(["pdf", "tsv", "csv", "txt", "tab", "xlsx", "xls"]),
        accept_multiple_files=True,
    )
    if len(uploaded_files) == 0:
        return

    with st.spinner("Documents are being processed"):
        chunked_documents = prepare_documents(uploaded_files)
        # chunked_documents = load_multiple_documents_and_split(temp_files)
        retriever = create_vector_store_retriever(
            model_host, chunked_documents
        )

    with st.spinner("ChatBot is being prepared"):
        llm = create_llm(model)
        prompt_template = create_main_prompt()
        qa = create_retrieval_qa(llm, prompt_template, retriever)

    for user_message, assistant_message in st.session_state.messages.items():
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_message)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(assistant_message)
            # sound_file = BytesIO()
            # tts = gTTS(assistant_message, lang="tr")
            # tts.write_to_fp(sound_file)
            # st.audio(sound_file)

    text_input = st.chat_input(
        placeholder="Yazarak sorun ✍️",
        key="text_box",
        max_chars=100,
    )
    # voice_input = speech2text()
    # user_input = voice_input or text_input
    user_input = text_input

    try:
        if user_input:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            with st.spinner("Question is being answered"):
                response = qa.run(user_input)

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()

                llm_output = ""
                for i in range(len(response)):
                    llm_output += response[i]
                    message_placeholder.write(f"{llm_output}▌")
                    time.sleep(STREAMING_INTERVAL)
                message_placeholder.write(llm_output)
                # sound_file = BytesIO()
                # tts = gTTS(llm_output, lang="tr")
                # tts.write_to_fp(sound_file)
                # st.audio(sound_file)

            if user_input not in st.session_state.messages:
                assistant_message = llm_output
                st.session_state.messages[user_input] = assistant_message

    except Exception as e:
        _, center_err_col, _ = st.columns([1, 8, 1])
        center_err_col.error(
            "\n Your question could not be answered. Please try to ask another question. Thank you ;]"
        )
        print(f"An error occurred: {type(e).__name__}")
        print(e)


if __name__ == "__main__":
    main()
