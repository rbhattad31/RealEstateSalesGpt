from langchain import FAISS, GoogleSerperAPIWrapper, SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import AzureOpenAI
from langchain.document_loaders import TextLoader
from pydantic import BaseModel, Field, validator
from loguru import logger
from Real_estate.logger import time_logger


@time_logger
def add_knowledge_base_products_to_cache(product_catalog: str = None):
    """
        We assume that the product catalog is simply a text string.
        """
    # load the document and split it into chunks
    logger.info("Inside Add Knowledge Base")
    loader = TextLoader(product_catalog, encoding='utf8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(deployment="bradsol-embedding-test",chunk_size=1)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")


def setup_knowledge_base(product_catalog: str = None):
    print("Inside Set Up Knowledge Base")
    """
    We assume that the product catalog is simply a text string.
    """
    llm = AzureOpenAI(temperature=0.2, deployment_name="qnagpt5", model_name="gpt-35-turbo")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(deployment="bradsol-embedding-test")
    db = FAISS.load_local("faiss_index", embeddings)
    knowledge_base = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return knowledge_base


def get_tools(knowledge_base):
    # we only use one tool for now, but this is highly extensible!
    search = SerpAPIWrapper()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer property information like PropertyImage, FlatImage, ApartmentImage,Flats,Villa,Property Type,Price,City,Community,Sub Community,Title,Amenities,Size,Bedrooms,image and questions related to property",
        ),
        Tool(
            name="NeighbourhoodSearch",
            func=wikipedia.run,
            description="useful for when you need to answer questions regards property neighbourhood details and features nearby property"
        )
    ]

    return tools
