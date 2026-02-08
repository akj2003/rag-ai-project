import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP: Define your models
print("--- 1. Initializing AI Models ---")
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. INGESTION: Load and split the PDF
pdf_path = "my_doc.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ Error: Could not find '{pdf_path}'. Please add a PDF file to this folder.")
    exit()

print(f"--- 2. Loading '{pdf_path}' ---")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"✅ Split document into {len(splits)} chunks.")

# 3. STORAGE: Create the Vector Database
print("--- 3. Creating Vector Database ---")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 4. MODERN CHAIN: The New Way (Replaces RetrievalQA)
# Create a prompt template that tells the AI strictly what to do
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Build the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. INTERACTION: Chat Loop
print("\n🤖 RAG System Ready! Ask a question about your PDF (or type 'exit' to quit):")
while True:
    query = input("\n👉 Question: ")
    if query.lower() == 'exit':
        break
    
    # Run the query
    response = rag_chain.invoke({"input": query})
    print(f"🤖 Answer: {response['answer']}")

# Cleanup
vectorstore.delete_collection()