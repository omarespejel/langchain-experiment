# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from getpass import getpass
# import os

# # from transformers import GPT2TokenizerFast

# loader = PyPDFLoader("./cairo_whitepaper.pdf")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 1000,
#     chunk_overlap  = 20,
# )
# texts = text_splitter.split_documents(data)

# print(f"There are {len(texts)} documents in the after splitting. Before splitting there was {len(data)} documents.")
# print(f"The number of characters in the first split document is {len(texts[0].page_content)}. Before splitting it was {len(data[0].page_content)}.")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# print(f"The type of the embeddings is {type(embeddings)}. The model name is {embeddings.model_name}.")
# # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# query_result = embeddings.embed_query("This is a test document.")
# print(f"The type of the query result is {type(query_result)}.")
# print(f"The type of the first element in the query result is {type(query_result[0])}.")
# print(f"The shape of the whole query result is {len(query_result)}.")

# llm = OpenAI("text-davinci-003", temperature=0)
# # HUGGINGFACEHUB_API_TOKEN = getpass()
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# chain = load_qa_chain(llm, chain_type="stuff")

# faiss_index = FAISS.from_documents(texts, embeddings)
# query = "How does Cairo compares to the EVM?"
# docs = faiss_index.similarity_search(query)

# print(chain.run(input_documents=docs, question=query))


import os
from getpass import getpass
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def test(assertion, expected, message):
    if assertion != expected:
        print(f"❌ {message}")
    else:
        print(f"✅ {message}")

try:
    loader = PyPDFLoader("./cairo_whitepaper.pdf")
    data = loader.load()
except FileNotFoundError:
    print("File not found. Please provide a valid file path.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
texts = text_splitter.split_documents(data)

print(f"There are {len(texts)} documents in the after splitting. Before splitting there was {len(data)} documents.")
print(f"The number of characters in the first split document is {len(texts[0].page_content)}. Before splitting it was {len(data[0].page_content)}.")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
test(type(embeddings), HuggingFaceEmbeddings, "Type of embeddings")
test(embeddings.model_name, "all-MiniLM-L6-v2", "Model name")

query_result = embeddings.embed_query("This is a test document.")
test(type(query_result), list, "Type of query result")
test(type(query_result[0]), float, "Type of the first element in the query result")
test(len(query_result), 384, "Shape of the whole query result")

llm = OpenAI(temperature=0)

chain = load_qa_chain(llm, chain_type="stuff")

faiss_index = FAISS.from_documents(texts, embeddings)
query = "How does Cairo compares to the EVM?"
docs = faiss_index.similarity_search(query)

print(chain.run(input_documents=docs, question=query))
