import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma

#azure openai model config
config_list = {
    "gpt-4o": [
        {
           "model": "gpt-4o",
            "api_key": "b3e1b40501854d7e8cd753dbfb97487c",
            "azure_endpoint": "https://aoai-apis.xmindai.cn/openai/deployments/xchat4o/chat/completions?api-version=2024-06-01",
            "api_type": "azure",
            "api_version": "2024-06-01",
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

os.environ["AZURE_OPENAI_API_KEY"] = "b3e1b40501854d7e8cd753dbfb97487c"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoai-apis.xmindai.cn/openai/deployments/xembedding3l/embeddings?api-version=2024-02-01"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
os.environ["DEPLOYMENT_NAME"] = "text-embedding-3-large"

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
vector_store = Chroma(
    collection_name="tools_api", #name of the collection, modified as needed
    embedding_function=embeddings,
    persist_directory="/data/yangliu/agent-benchmark/bioagent/vector_store/" # path to your vector store directory, modified as needed
)