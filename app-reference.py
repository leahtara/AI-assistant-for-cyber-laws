from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

# Load and preprocess data
def load_and_chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

# Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

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

# Streamlit UI
def main():
    st.title("Indian Cyber Laws AI Assistant")
    query = st.text_input("Ask a question about Indian Cyber Laws:")

    if query:
        # Load and preprocess data
        chunks = load_and_chunk_pdf("data/cyber_laws.pdf")
        
        # Create vector store
        vector_store = create_vector_store(chunks)
        
        # Set up RAG pipeline
        qa_pipeline = setup_rag_pipeline(vector_store)
        
        # Get response
        response = qa_pipeline.run(query)
        st.write("Response:")
        st.write(response)

if __name__ == "__main__":
    main()