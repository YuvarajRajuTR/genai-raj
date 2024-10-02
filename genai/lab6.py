import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama 
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Configuration
PDF_DIR = "/workspaces/rag/data/"  # Update this path
CHROMA_DATA_PATH = "vdb_data/"

# 1. Extract data
def load_pdf_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# 2. Split into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# 3. Create embeddings and 4. Build semantic index
def create_vector_store(texts):
    embeddings = FastEmbedEmbeddings()
    db_chroma = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DATA_PATH)
    db_chroma.persist()
    return db_chroma

# Load and process documents
pdf_documents = load_pdf_documents(PDF_DIR)
chunks = split_documents(pdf_documents)
vector_store = create_vector_store(chunks)

# Initialize Ollama model
llm = Ollama(model="llama3", temperature=0)

# Set up RetrievalQA
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Define prompt template
PROMPT_TEMPLATE = """
Answer the question: {question} using whatever resources you have.
Include any related information from {context} as part of your answer.
Provide a detailed answer.
Don't justify your answers.
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Set up RetrievalQA with the new prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Main interaction loop
print("RAG Pipeline with PDF documents")
print("Ask questions about the content of the loaded PDFs.")
print("Type 'exit' to quit.")

while True:
    query = input("\nQuery: ")
    if query.lower() == "exit":
        break
    if query.strip() == "":
        continue
    
    # Use the RetrievalQA chain to get the answer
    result = qa_chain({"query": query})
    
    print("\nResponse:", result['result'])
    print("\nSources:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown source')}")

print("Thank you for using the RAG Pipeline. Goodbye!")