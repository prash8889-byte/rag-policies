from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, ChatPromptTemplate


loader = TextLoader('policies.txt')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embedding_model = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="policies",
    persist_directory="./policy_chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

model = ChatNVIDIA(model="nvidia/nemotron-3b-chat", temperature=0.2, max_output_tokens=512)

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions about company policies. " \
    "Use the following context to answer the question at the end. " \
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. " \
    "Context: {context} " \
    "Question: {question}"
)

question = input("What is your question about company policies? ")

retrieved_docs = retriever.invoke(question)

context = "\n".join([doc.page_content for doc in retrieved_docs])

print(context)