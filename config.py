"""
EXAMPLE
Configuration file for Ollama tool functions.
Contains all settings, API keys, and tool definitions.
"""

# Ollama Configuration
OLLAMA_CONFIG = {
    "base_url": "http://your-ollama-url:11434",  # Replace with your Ollama URL
    "default_model": "llama3.1",
    "timeout": 30
}

# API Keys
API_KEYS = {
    "openweather": "your_openweather_api_key",  # Get from https://openweathermap.org/api
    "alpha_vantage": "your_alpha_vantage_api_key",  # Get from https://www.alphavantage.co/
    "news_api": "your_news_api_key"  # Get from https://newsapi.org/
}

# Model Parameters
MODEL_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_ctx": 4096,
    "repeat_penalty": 1.1,
    "num_predict": 256
}

# Tool Function Definitions
TOOL_FUNCTIONS = {
    "get_current_weather": {
        "name": "get_current_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city",
                }
            },
            "required": ["city"]
        }
    },
    "search_web": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    "analyze_text": {
        "name": "analyze_text",
        "description": "Analyze text and provide insights",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze"
                }
            },
            "required": ["text"]
        }
    },
    "get_stock_data": {
        "name": "get_stock_data",
        "description": "Get stock market data for a symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol (e.g., AAPL, GOOGL)"
                }
            },
            "required": ["symbol"]
        }
    },
    "get_forex_data": {
        "name": "get_forex_data",
        "description": "Get forex exchange rate data",
        "parameters": {
            "type": "object",
            "properties": {
                "from_symbol": {
                    "type": "string",
                    "description": "The source currency symbol (e.g., USD, EUR)"
                },
                "to_symbol": {
                    "type": "string",
                    "description": "The target currency symbol (e.g., USD, EUR)"
                }
            },
            "required": ["from_symbol", "to_symbol"]
        }
    },
    "analyze_financial_data": {
        "name": "analyze_financial_data",
        "description": "Analyze financial data and provide insights",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The financial data to analyze"
                }
            },
            "required": ["data"]
        }
    },
    "get_news": {
        "name": "get_news",
        "description": "Get recent news articles about a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for news"
                }
            },
            "required": ["query"]
        }
    },
    "analyze_sentiment": {
        "name": "analyze_sentiment",
        "description": "Analyze the sentiment of text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze"
                }
            },
            "required": ["text"]
        }
    },
    "summarize_text": {
        "name": "summarize_text",
        "description": "Summarize text concisely",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                }
            },
            "required": ["text"]
        }
    },
    "get_population": {
        "name": "get_population",
        "description": "Get population data for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                }
            },
            "required": ["city"]
        }
    },
    "search_database": {
        "name": "search_database",
        "description": "Search for products in the database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    "get_product_details": {
        "name": "get_product_details",
        "description": "Get details about a specific product",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product identifier"
                }
            },
            "required": ["product_id"]
        }
    }
}

# API Endpoints
API_ENDPOINTS = {
    "openweather": "http://api.openweathermap.org/data/2.5/weather",
    "alpha_vantage": "https://www.alphavantage.co/query",
    "news_api": "https://newsapi.org/v2/everything",
    "duckduckgo": "https://api.duckduckgo.com/"
}

# Error Messages
ERROR_MESSAGES = {
    "missing_api_key": "API key for {service} is not configured. Please add it to config.py",
    "connection_error": "Failed to connect to {service}. Please check your internet connection.",
    "invalid_response": "Received invalid response from {service}. Please try again later.",
    "rate_limit": "Rate limit exceeded for {service}. Please wait before making more requests."
} 
