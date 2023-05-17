# Telnyx QA ChatBot

This is a chatbot that answers questions based on SMS guidelines documentation from Telnyx. You can use it via the command line.


## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)


## Prerequisites

Before setting up the project, ensure that you have the following prerequisites installed on your machine:

- Python (version 3.10.10):
(https://www.python.org/downloads/)
- Poetry (version 1.4.0):
(https://python-poetry.org/docs/#installation)


## Installation

1. Clone the project repository:

```
git clone https://github.com/olympian-21/Simple-Chatbot.git
```

2. Navigate to the project directory:

```
cd Telnyx-QA-ChatBot
```

3. Install project dependencies using Poetry:

```
poetry install
```

This command will create a virtual environment and install all the necessary dependencies specified in the `pyproject.toml` file.


## Usage

1. Activate the virtual environment created by Poetry:

```
poetry shell
```

2. Run the project:

```
python src/app.py "query (your question)"
```


## Challenges during the development

1. The OPEN-AI token limit was a problem when used with charactertextsplitter. Thus, a markdown text splitter with the appropriate chunk_size parameter is used.
2: Different chain_type parameters are used in the retrival_qa object. The best chain type is found to be stuff in terms of accuracy.
Chromadb did not persist at first, but later, I learned about the necessity of the persist method.
