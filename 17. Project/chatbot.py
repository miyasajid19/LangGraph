from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings,ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA




DATA_PATH = r"E:/LangGraph/17. Project/books"
OLLAMA_MODEL="deepseek-v3.1:671b-cloud"
OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
DB_FAISS_PATH="E:/LangGraph/17. Project/vectorstore/db_faiss"


load_dotenv()


# ------------
# loading PDF documents
# ------------
def load_pdf(file_path):
    loader=DirectoryLoader(
        path=file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def load_all(file_path):
    loader=DirectoryLoader(
        path=file_path,
        glob="**/*.*",  # Recursive pattern to search all subfolders
        recursive=True,
        silent_errors=True,  # Skip files that can't be loaded
        show_progress=True   # Show progress bar
    )
    documents = loader.load()
    return documents

# ------------
# splitting documents into chunks
# ------------
def split_docs(documents):
    chunks = []
    
    for doc in documents:
        print(f"Processing document: {doc.metadata['source']}")
        ext = os.path.splitext(doc.metadata['source'])[1].lower()
        
        # Determine splitter based on file type
        if ext == '.py':
            print("Using Python text splitter")
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=500,
                chunk_overlap=100,
            )
        elif ext in ['.cpp', '.h']:
            print("Using C++ text splitter")
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP,
                chunk_size=500,
                chunk_overlap=100,
            )
        elif ext == '.ipynb':
            print("Using Jupyter text splitter")
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.JUPYTER,
                chunk_size=500,
                chunk_overlap=100,
            )
        else:  # PDF and others
            print("Using default text splitter")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
        
        # Split and accumulate chunks
        chunks.extend(text_splitter.split_documents([doc]))
    
    return chunks
        

# ------------
# creating embedder
# ------------
def get_embeddings():
    embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL,
    )
    return embeddings


# ------------
# creating vector store
# ------------
def create_vector_store(chunks, embeddings):
    if not chunks:
        raise ValueError("No documents found to create vector store. Please check your data path and file types.")
    
    db=FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    


#  ----------
# getting llm
# ----------
def get_llm():
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.7,
        max_tokens=2048
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt



# ------------
# getting vector database
# ------------
def get_vector_store():
    # if not exist create it 
    if not os.path.exists(DB_FAISS_PATH):
        print("Creating vector store...")
        documents = load_all(DATA_PATH)
        if not documents:
            raise ValueError(f"No documents found in {DATA_PATH}. Please check the path.")
        print(f"Loaded {len(documents)} documents")
        chunks = split_docs(documents)
        print(f"Created {len(chunks)} chunks")
        create_vector_store(chunks, get_embeddings())
        print("Vector store created and saved.")
    
    db=FAISS.load_local(DB_FAISS_PATH, get_embeddings(),allow_dangerous_deserialization=True)
    return db


# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=get_llm(),
    chain_type="stuff",
    retriever=get_vector_store().as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

if __name__ == "__main__":
    print("Chatbot is ready to answer your questions!")
    while True:
        user_question = input("\nUser:  ")
        if user_question.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        response = qa_chain.invoke(user_question)
        print(f"\nBot: {response['result']}")