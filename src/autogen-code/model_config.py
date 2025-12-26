import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma

#azure openai model config
config_list = {
    "gpt-4o": [
        {
           "model": "gpt-4o",
            "api_key": "xxx",
            "azure_endpoint": "xxxx",
            "api_type": "azure",
            "api_version": "xxx",
        }
    ]
} 

#openai model config
'''
config_list = {
    "gpt-4o": [
        {
        "api_type": "openai",
        "base_url": "xxx",
        "api_key": "xxx",
        "model": "xxx",
        }
    ]
    "qwen-max": [
        {
        "api_type": "openai",
        "base_url": "xxx",
        "api_key": "xxx",
        "model": "xxx",
        }
    ]
    ...
}
'''

os.environ["AZURE_OPENAI_API_KEY"] = "xxxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "xxxx"
os.environ["AZURE_OPENAI_API_VERSION"] = "xxxx"
os.environ["DEPLOYMENT_NAME"] = "xxxx"

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
vector_store = Chroma(
    collection_name="tools_api", #name of the collection, modified as needed
    embedding_function=embeddings,
    persist_directory="/data/yangliu/agent-benchmark/vector_store/" # path to your vector store directory, modified as needed
)