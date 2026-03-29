# 🧬 BioMed-RAG: A Local AI Assistant for Medical Texts

Hi! I'm a Biomedical Engineering student expanding my skill set into Data Engineering and Generative AI. I built this proof-of-concept project to get hands-on experience with RAG (Retrieval-Augmented Generation) pipelines, vector databases, and document processing.

My goal was to build a tool that can ingest dense academic sources and accurately answer questions based on that text. To prioritize data privacy and keep costs at zero, I designed this system to run 100% locally using open-source models, without relying on external or paid APIs.

## 🛠️ What This Code Does
It is a Python command-line application that executes the following workflow:

1. **Ingests Data:** Reads PDFs from a local `/sources/` folder using LangChain.

2. **Chunks Text:** Breaks the document into 1,000-character chunks with a 200-character overlap to preserve context.

3. **Creates Embeddings:** Uses Ollama's `nomic-embed-text` to turn those chunks into vectors and stores them persistently in a local Chroma database.

4. **Answers Questions:** Takes user input from the terminal, retrieves the most relevant textbook paragraphs from Chroma, and feeds them to a local `llama3` model to generate a concise, context-aware answer.

## 💻 Tech Stack

* **Python** * **LangChain** (Document loaders, text splitters, and retrieval chains)
* **ChromaDB** (Local vector storage)
* **Ollama** (Local LLM engine)

## 🚀 How to Run It Locally

1. **Get Ollama and the models:**

   Download Ollama from [ollama.com](https://ollama.com/), open your terminal, and pull the two models I used:
   `ollama pull nomic-embed-text`
   `ollama pull llama3`

2. **Set up the Python environment:**
   `python -m venv venv`
   `source venv/bin/activate`  *(Or `venv\Scripts\activate` on Windows)*
   `pip install langchain_community langchain_text_splitters langchain_ollama langchain_chroma langchain_classic pypdf`

3. **Add your PDFs:**
   Create a folder named `sources` in the same directory as the script and drop a PDF in there (I tested this with a 700+ page molecular biology textbook).

4. **Run the script:**
   `python bioRAG.py`

   *Note: The first time you run it, it might take a few minutes to chunk and embed a large PDF into the Chroma database. After that, you can query the database instantly via the terminal loop.*

## 🌱 What I Learned & Future Ideas

Building this helped me practically understand the end-to-end pipeline of an AI data application. If I were to expand this in the future, I would:

* Add UI for easier usage
* Expand the database
* Add a memory buffer to allow the application to remember previous questions.