from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader("data.pdf")
    docs = loader.load()
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    
    return split_docs

def create_vector_store(docs):
    texts = [doc.page_content for doc in docs]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_texts(texts, embedding=embeddings)
    return store
