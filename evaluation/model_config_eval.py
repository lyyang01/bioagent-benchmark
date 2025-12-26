CONFIG_LIST = {
        "gpt-4o": [{
            "model": "gpt-4o",
            "api_key": "b3e1b40501854d7e8cd753dbfb97487c",
            "azure_endpoint": "https://aoai-apis.xmindai.cn/openai/deployments/xchat4o/chat/completions?api-version=2024-06-01",
            "api_type": "azure",
            "api_version": "2024-06-01",
        }],

        "deepseek-r1": [
            #{
            #    "api_key":"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",
            #    "model": "deepseek/deepseek-r1",
            #    "base_url": "https://openrouter.ai/api/v1",
            #    "api_type": "openai",
            #},
           {
            "model": "deepseek-r1",
            "api_key": "b3e1b40501854d7e8cd753dbfb97487c",
            "azure_endpoint": "https://aoai-apis.xmindai.cn/openai/deployments/xr1pro/chat/completions?api-version=2024-06-01",
            "api_type": "azure",
            "api_version": "2024-06-01",
        }],

        "grok-4": 
        [{
            "model": "grok-4",
            "api_key": "b3e1b40501854d7e8cd753dbfb97487c",
            "azure_endpoint": "https://aoai-apis.xmindai.cn/openai/deployments/xgrok4/chat/completions?api-version=2025-04-01-preview",
            "api_type": "azure",
            "api_version": "2025-04-01-preview",
        }],

        "gemini-2.5-pro": 
        [{   
            "api_version": "2024-10-21",
            "azure_endpoint": "https://apis.openroutex.com/openai/deployments/xgemini25pro/chat/completions?api-version=2024-10-21",
            "api_key": "b3e1b40501854d7e8cd753dbfb97487c",
            "model": "genmini-2.5-pro",
            "api_type": "azure",        
        }],

        "gpt-4.1": 
        [  
            {   
                "model": "xchat41",
                "api_key": "b3e1b40501854d7e8cd753dbfb97487c",    
                "azure_endpoint": "https://aoai-apis.xmindai.cn",
                "api_version": "2024-08-01-preview",
                #"model": "gpt-4.1",
                #"api_key": "b3e1b40501854d7e8cd753dbfb97487c",
                #"azure_endpoint": "https://aoai-apis.xmindai.cn/openai/deployments/xchat41/chat/completions?api-version=2024-08-01-preview",
                #"api_type": "azure",
                #"api_version": "2024-08-01-preview",
            },
        ]
}