import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

os.environ["AZURE_OPENAI_API_VERSION"] = "xxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "xxx"
os.environ["AZURE_OPENAI_API_KEY"] = "xxx"
os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT_NAME"] = "xxx"

embeddings = AzureOpenAIEmbeddings(
    # azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

vector_store = Chroma(
    collection_name="tools_api",
    embedding_function=embeddings,
    persist_directory="../vector_store"
    )

