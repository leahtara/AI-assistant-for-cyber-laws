# import langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# I have no idea what this is for yet, let's see if it works
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# streamlit for UI
import streamlit as st

# os so that we don't have to create that rag every time
import os

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "./vector_store.faiss"  # Path to save/load the vector store


# Load and preprocess data
def load_and_chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

# Create or load vector store
def get_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing vector store
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    else:
        # Create new vector store
        chunks = load_and_chunk_pdf("data/cyber_laws.pdf")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)  # Save for future use
        return vector_store

# Load LLaMA 2 model
def load_llama2_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=pipe)

# Set up RAG pipeline
def setup_rag_pipeline(vector_store):
    llm = load_llama2_model()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )


def main(): # basically the UI and calling all the functions

    # app title
    st.title("ClAI")

    # Initialize vector store and RAG pipeline
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()
    if "qa_pipeline" not in st.session_state:
        st.session_state.qa_pipeline = setup_rag_pipeline(st.session_state.vector_store)

    # storing old prompts for conversation in a state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # diplay all the historical messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # prompt input
    prompt = st.chat_input("Ask me about Indian Cyber Laws")

    if prompt:
        # display the prompt
        st.chat_message('user').markdown(prompt)
        # store the user prompts the state
        st.session_state.messages.append({'role':'user', 'content':prompt})
        # send prompt to llm
        response = st.session_state.qa_pipeline.run(prompt)
        # show llm response
        st.chat_message('assistant').markdown(response)
        # store the llm response in state also for displaying
        st.session_state.messages.append({'role':'assistant', 'content':response})


if __name__ == "__main__":
    main()
