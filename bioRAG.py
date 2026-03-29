from langchain_community.document_loaders import PyPDFDirectoryLoader

#loading the sources

loader= PyPDFDirectoryLoader('./sources/', glob="**/*.pdf")

docs=loader.load()

# pypdfloader loads one documnent object per pdf page, so lets check how many objectsdd we go
print(len(docs))



# splitting the documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #overlap to prevent loosing context mid sentence

texts=text_splitter.split_documents(docs)



# embedding the chunks fo ollama 

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")




# storing the embedded chunks in chroma
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="biomed_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

ids = vector_store.add_documents(documents=texts)

# quering the database




# setting up LLM response

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain

llm=OllamaLLM(model="llama3")

template = """ You are a biomedical assistant that answers user questions based on provided sources. If you cant find the answer, say that you don't know.
Context: {context}

Question: {input}

Answer:
"""

prompt = PromptTemplate.from_template(template)

combine_docs_chain=create_stuff_documents_chain(llm, prompt)
retriever= vector_store.as_retriever()
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# creating a loop to be able to ask questions

print("\n The application is ready. Type exit or quit to end the program. You can now type your question")

while True:

    question = input("\n Type your question here: ")

    if question.lower() in ['exit', 'quit']:
        print("Program ended")
        break

    # just in case the user clicks enter without typing anything
    if not question.strip():
        continue

    print("Processing your question...")

    response= rag_chain.invoke({"input": question})

    print(response["answer"])

    print("-" *50)