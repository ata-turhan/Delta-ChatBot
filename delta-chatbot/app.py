import os

import streamlit as st
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.llms import OpenAI
from modules.utils import add_bg_from_local, set_page_config

# Storing the chat
if "user" not in st.session_state:
    st.session_state.user = []
if "bot" not in st.session_state:
    st.session_state.bot = []
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "model" not in st.session_state:
    st.session_state.model = None


def start_chat():
    # Create instance of OpenAI LLM
    if st.session_state.model.startswith("openai"):
        llm = OpenAI(temperature=0.1, verbose=True)
    else:
        llm = HuggingFaceHub(
            repo_id=st.session_state.model,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 512,
            },
        )
    question = st.text_input("Please write your question:")

    template = """Question: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    st.write(llm_chain.run(question))


def is_api_key_valid(model: str, api_key: str):
    if api_key is None:
        st.sidebar.warning("Please enter a valid API Key!", icon="⚠")
        return False
    elif model == "openai" and not api_key.startswith("sk-"):
        st.sidebar.warning("Please enter a valid API Key!", icon="⚠")
        return False
    elif model == "huggingface" and not api_key.startswith("hf_"):
        st.sidebar.warning("Please enter a valid API Key!", icon="⚠")
        return False
    else:
        key = (
            "OPENAI_API_KEY"
            if model == "openai"
            else "HUGGINGFACEHUB_API_TOKEN"
        )
        os.environ[key] = api_key
        return True


def show_sidebar():
    st.sidebar.markdown(
        "<center><h1>Configurations</h1></center> <br>",
        unsafe_allow_html=True,
    )

    llm = st.sidebar.selectbox(
        "Please select a LLM:",
        [
            "<Select>",
            "openai/gpt-3.5-turbo",
            "google/flan-t5-xxl",
            "databricks/dolly-v2-3b",
            "Writer/camel-5b-hf",
            "Salesforce/xgen-7b-8k-base",
            "tiiuae/falcon-40b",
            "bigscience/bloom",
        ],
    )
    st.session_state.model = llm
    if llm != "<Select>":
        st.sidebar.text_input(
            f"Please enter the {llm} API Key:", key="api_key"
        )
        model = "openai" if llm.startswith("openai") else "huggingface"
        if is_api_key_valid(model, st.session_state.api_key):
            st.sidebar.success("API Key was received successfully.")
            start_chat()


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
        """<h1 style='text-align: center; color: black; font-size: 60px;'> 📝 Delta - Document ChatBot </h1>
        <br>""",
        unsafe_allow_html=True,
    )

    show_sidebar()


if __name__ == "__main__":
    main()
