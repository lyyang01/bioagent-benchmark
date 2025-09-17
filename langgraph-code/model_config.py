import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

LLM_CONFIG = {
   "gpt-4.1": {
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
   },
   "gpt-4o": {
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
   },
    "deepseek-r1": {
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
   },
    "deepseek-v3":{
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
    },
    
    "qwen-max":{
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
    },

    "sonnet-3.7":{
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
    },
    "genmini-2.5-pro":{
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
    },
    "grok3-beta":{
       "api_version": "xxxx",
       "base_url": "xxx",
       "api_key": "xxx",
       "model": "xxx",
    }
}



class LLMFactory:
    """LLM factory class, supporting Azure OpenAI, OpenAI, Anthropic Claude, and extendable to more models"""
    _instance = None
    @classmethod
    def create_model(cls, model, **kwargs):
        """
        Create LLM instance based on provider.

        :param model: LLM name (gpt-4o, deepseek-r1, etc.)
        :param kwargs: Additional parameters (e.g., temperature, streaming, etc.)
        :return: LLM instance
        """
        config = LLM_CONFIG[model]
        if model == "gpt-4o" or model =="gpt-4.1" or model =="deepseek-r1":
            cls._instance = AzureChatOpenAI(
                azure_endpoint=config["base_url"],
                api_key=config["api_key"],
                api_version=config["api_version"],
                azure_deployment=config["model"],
                temperature=0, 
                # streaming=True,
                # response_format=None,
                # max_tokens=5000
            )
        else:
            cls._instance = ChatOpenAI(
                    openai_api_key=config["api_key"],
                    openai_api_base=config["base_url"],
                    model_name=config["model"],
                    temperature=0, 
                    )
        return cls._instance
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("LLM not initialized")
        return cls._instance

def initialize_llm(model):
    return LLMFactory.create_model(model)
