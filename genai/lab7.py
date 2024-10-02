import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Set some config variables for ChromaDB
CHROMA_DATA_PATH = "data_classification_vdb/"
DOCS_PATH = "/workspaces/rag/data/"  # Directory containing your policy PDFs

# Initialize Ollama model
llm = Ollama(model="llama3")

# Load and process all PDF documents in the specified directory
def load_and_process_documents(directory):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Split the doc into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            doc_chunks = text_splitter.split_documents(pages)
            chunks.extend(doc_chunks)
    return chunks

# Load and process all documents
all_chunks = load_and_process_documents(DOCS_PATH)

# Create embeddings
embeddings = FastEmbedEmbeddings()

# Embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(all_chunks, embeddings, persist_directory=CHROMA_DATA_PATH)

PROMPT_TEMPLATE = """
You are an AI assistant specializing in data classification and policy implementation.
Answer the question: {question} based on the company's data classification policies and guidelines.
Use the following context information to inform your answer: {context}
Provide a detailed and specific answer that helps the user understand how to properly classify data or implement policies.
If the information is not available in the context, say so and provide general best practices.
"""

print("Data Classification and Policy Assistant")
print("Ask questions about data classification or policy implementation.")
print("Type 'exit' to quit.")

while True:
    query = input("\nQuery: ")
    if query.lower() == "exit":
        break
    if query.strip() == "":
        continue

    # Retrieve context - top 5 most relevant chunks to the query vector
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

    # Generate an answer based on given user query and retrieved context information
    context_text = "\n\n".join([doc.page_content for doc, score in docs_chroma])

    # Use the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Call LLM model to generate the answer based on the given context and query
    response_text = llm.invoke(prompt)
    print("\nResponse:", response_text)

    # Optionally, print sources (you may want to format this better)
    print("\nSources:")
    for i, (doc, score) in enumerate(docs_chroma, 1):
        print(f"{i}. (Score: {score:.2f}) {doc.page_content[:100]}...")