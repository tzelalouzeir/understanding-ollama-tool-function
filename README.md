# Ollama Tool Functions Implementation Guide

This guide explains how to implement and use tool functions with Ollama's LLM models.

## Prerequisites

1. Ollama installed and running
2. Python 3.7+
3. Required Python packages:
```bash
pip install ollama requests pandas yfinance nltk
```

## Configuration

Before using the examples, make sure to:

1. Create a `config.py` file with your settings:
```python
from config import OLLAMA_CONFIG, API_KEYS, MODEL_PARAMS, TOOL_FUNCTIONS, API_ENDPOINTS, ERROR_MESSAGES
```

2. Update the configuration with your actual values:
   - Set your Ollama server URL in `OLLAMA_CONFIG["base_url"]`
   - Add your API keys in `API_KEYS`
   - Adjust model parameters in `MODEL_PARAMS` if needed

## Basic Example

```python
from ollama import Client
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Configure Ollama using environment variables
# For Mac: launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
# For Linux: Add Environment="OLLAMA_HOST=0.0.0.0:11434" in systemd service
# For Windows: Set OLLAMA_HOST in system environment variables
# 
# Alternative: Configure the API client with the base URL directly
client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

response = client.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{
      'type': 'function',
      'function': TOOL_FUNCTIONS["get_current_weather"]
    }],
)

print(response['message']['tool_calls'])
```

> [!NOTE]
> Make sure to update the `config.py` file with your actual Ollama server URL and API keys before running the examples.

# Understanding Ollama's Tool Function Implementation

## Table of Contents

1. [Introduction](#introduction)
2. [Ollama's Architecture for Tool Functions](#ollamas-architecture-for-tool-functions)
3. [Template System in Ollama](#template-system-in-ollama)
4. [Tool Function Implementation Details](#tool-function-implementation-details)
5. [How LLMs Generate Tool Calls](#how-llms-generate-tool-calls)
6. [Parsing Tool Calls from LLM Output](#parsing-tool-calls-from-llm-output)
7. [API Integration](#api-integration)
8. [Practical Examples](#practical-examples)
9. [Tool Execution Workflow](#tool-execution-workflow)
10. [Multiple Tool Usage](#multiple-tool-usage)
11. [Hierarchical Agent Tool Usage](#hierarchical-agent-tool-usage)
12. [Real Executable Examples](#real-executable-examples)
13. [Conclusion](#conclusion)

## Introduction

Ollama is an open-source framework that allows running large language models (LLMs) locally. One of its advanced features is the support for tool functions, which enables LLMs to interact with external tools and APIs to perform tasks beyond text generation.

This document explores how Ollama implements tool functions, from the template system that structures prompts to the parsing mechanism that extracts tool calls from model outputs.

## Ollama's Architecture for Tool Functions

Ollama's tool function capability is built on several key components:

1. **API Types**: Definitions of tool-related structures (in `api/types.go`)
2. **Template System**: Customizable prompt templates that include tool call support (in `template` package)
3. **Parsing Logic**: Code that extracts tool calls from model outputs (in `server/model.go`)
4. **API Integration**: OpenAI-compatible endpoints for tools (in `openai/openai.go`)

The architecture follows a modular design where:
- Model templates determine how tool prompts are formatted
- The server processes LLM responses to identify and extract tool calls
- The API layer provides a standardized interface for clients

## Template System in Ollama

Ollama uses Go's template system to format prompts for different LLMs. Templates are key to tool function support because they determine how tool definitions are presented to the model.

Templates in Ollama:
- Are defined using Go's `text/template` package
- Can include conditionals like `{{ if .System }}` to handle optional components
- Support special variables like `.ToolCalls` for tool integration

An example template that supports tools might look like:

```go
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}{{ if .ToolCalls }}{{ range .ToolCalls }}
<|im_start|>tool
{{ .Function.Name }}({{ .Function.Arguments }})
<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
{{ .Response }}"""
```

Templates are stored in the model's metadata and used during inference to structure prompts properly.

## Tool Function Implementation Details

Tool functions in Ollama are represented through several key structures:

```go
type Tool struct {
    Type     string       `json:"type"`
    Function ToolFunction `json:"function"`
}

type ToolFunction struct {
    Name        string `json:"name"`
    Description string `json:"description"`
    Parameters  struct {
        Type       string   `json:"type"`
        Required   []string `json:"required"`
        Properties map[string]struct {
            Type        string   `json:"type"`
            Description string   `json:"description"`
            Enum        []string `json:"enum,omitempty"`
        } `json:"properties"`
    } `json:"parameters"`
}

type ToolCall struct {
    Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
    Name      string                    `json:"name"`
    Arguments ToolCallFunctionArguments `json:"arguments"`
}

type ToolCallFunctionArguments map[string]any
```

These structures allow for:
1. Defining available tools that a model can use
2. Representing tool calls made by the model
3. Passing arguments to those tools

## How LLMs Generate Tool Calls

When an LLM is prompted with tool definitions, it needs to generate structured output that indicates which tool to call and with what parameters. This happens through:

1. **Prompt Engineering**: The template formats the prompt to instruct the model on available tools
2. **JSON Generation**: The model generates a JSON-formatted response containing tool calls
3. **Output Formatting**: The model follows the specified template to structure its response

Models like Llama 3.1, Mistral Nemo, and Firefunction v2 are specifically trained to generate tool calls in a structured format that Ollama can parse.

## Parsing Tool Calls from LLM Output

The heart of Ollama's tool function implementation is the `parseToolCalls` method in `server/model.go`. This function extracts structured tool calls from the LLM's output text:

```go
// parseToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// mxyng: this only really works if the input contains tool calls in some JSON format
func (m *Model) parseToolCalls(s string) ([]api.ToolCall, bool) {
    // Step 1: Find the template subtree that handles tool calls
    tmpl := m.Template.Subtree(func(n parse.Node) bool {
        if t, ok := n.(*parse.RangeNode); ok {
            return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
        }
        return false
    })
    if tmpl == nil {
        return nil, false
    }

    // Step 2: Execute template with dummy data to understand its structure
    var b bytes.Buffer
    if err := tmpl.Execute(&b, map[string][]api.ToolCall{
        "ToolCalls": {
            {
                Function: api.ToolCallFunction{
                    Name: "@@name@@",
                    Arguments: api.ToolCallFunctionArguments{
                        "@@argument@@": 1,
                    },
                },
            },
        },
    }); err != nil {
        return nil, false
    }

    // Step 3: Analyze template objects to find field mappings
    templateObjects := parseObjects(b.String())
    if len(templateObjects) == 0 {
        return nil, false
    }
    
    // Step 4: Determine which fields correspond to name and arguments
    var name, arguments string
    for k, v := range templateObjects[0] {
        switch v.(type) {
        case string:
            name = k
        case map[string]any:
            arguments = k
        }
    }
    if name == "" || arguments == "" {
        return nil, false
    }

    // Step 5: Parse the actual model response
    responseObjects := parseObjects(s)
    if len(responseObjects) == 0 {
        return nil, false
    }

    // Step 6: Recursively collect all nested objects
    var collect func(any) []map[string]any
    collect = func(obj any) (all []map[string]any) {
        switch o := obj.(type) {
        case map[string]any:
            all = append(all, o)
            for _, v := range o {
                all = append(all, collect(v)...)
            }
        case []any:
            for _, v := range o {
                all = append(all, collect(v)...)
            }
        }
        return all
    }
    
    var objs []map[string]any
    for _, p := range responseObjects {
        objs = append(objs, collect(p)...)
    }

    // Step 7: Extract tool calls from matching objects
    var toolCalls []api.ToolCall
    for _, kv := range objs {
        n, nok := kv[name].(string)
        a, aok := kv[arguments].(map[string]any)
        if nok && aok {
            toolCalls = append(toolCalls, api.ToolCall{
                Function: api.ToolCallFunction{
                    Name: n,
                    Arguments: a,
                },
            })
        }
    }

    return toolCalls, len(toolCalls) > 0
}
```

This complex parsing process:

1. **Uses template analysis**: Determines how the model structures tool calls based on its template
2. **Applies pattern matching**: Identifies function names and argument maps
3. **Handles nested structures**: Recursively traverses complex JSON outputs
4. **Creates structured objects**: Converts model output into usable `ToolCall` structures

The parser is flexible enough to handle different model-specific formats, as long as they contain the essential components of tool calls (function name and arguments).

## API Integration

Ollama exposes tool function capability through its API, which includes OpenAI-compatible endpoints. The key components include:

1. **Request Handling**: Processing incoming requests with tool definitions
2. **Response Formatting**: Structuring responses to include tool calls
3. **Middleware**: Converting between Ollama's internal format and OpenAI-compatible format

The OpenAI compatibility layer is particularly important, allowing seamless integration with existing tools and libraries.

## Practical Examples

### Python Example

```python
from ollama import Client
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Create a client with the base URL configuration
client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

response = client.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{
      'type': 'function',
      'function': TOOL_FUNCTIONS["get_current_weather"]
    }],
)

print(response['message']['tool_calls'])
```

### OpenAI Compatibility Example

```python
import openai
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Configure the base URL for your Ollama instance
openai.base_url = f"{OLLAMA_CONFIG['base_url']}/v1"
openai.api_key = 'ollama'

response = openai.chat.completions.create(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{"role": "user", "content": "What's the weather in New York?"}],
    tools=[{
      "type": "function",
      "function": TOOL_FUNCTIONS["get_current_weather"]
    }],
)

print(response.choices[0].message.tool_calls)
```

> [!NOTE]
> Make sure to update the `config.py` file with your actual Ollama server URL and API keys before running the examples.

## Tool Execution Workflow

A critical aspect to understand about Ollama's tool function implementation is that **Ollama itself does not execute the tools**. Ollama only handles:

1. Formatting tool definitions into prompts
2. Parsing tool calls from model outputs
3. Providing these structured tool calls to the client application

The actual execution of tools is the responsibility of the client application. Here's the complete workflow for tool execution:

```
┌─────────────┐    1. Tool definitions    ┌────────┐    2. Tool prompt      ┌─────┐
│ Your App    │───────────────────────────▶ Ollama │───────────────────────▶ LLM │
└─────┬───────┘                           └────┬───┘                        └──┬──┘
      │                                        │                               │
      │                                        │     3. Raw text with tool     │
      │                                        │        call embedded          │
      │                                        │◀─────────────────────────────┘
      │      4. Parsed tool call              │
      │◀───────────────────────────────────────┘
      │
      │      5. Execute tool in your application
      ├─────────────────────┐
      │                     │
      │                     ▼
      │      6. Get tool result
      │◀────────────────────┘
      │
      │      7. Send result back to LLM
      ▼
┌─────────────┐    Tool result message    ┌────────┐       Format result       ┌─────┐
│ Your App    │───────────────────────────▶ Ollama │──────────────────────────▶ LLM │
└─────────────┘                           └────────┘                           └─────┘
```

### Implementation Steps

1. **Define Tools**: Your application defines tools and their parameters
2. **Send Request**: The tools are sent to Ollama along with your prompt
3. **Parse Response**: Ollama returns the model's structured tool call
4. **Execute Tool**: Your application executes the specified function with the provided arguments
5. **Return Results**: Send the tool's output back to Ollama as a message with role "tool"

### Example Execution Flow

```python
from ollama import Client
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# 1. Define your tool implementation in your application
def get_weather(city):
    # Your actual API call or function implementation
    return f"It's 22°C and sunny in {city}"

# 2. Create a client with the base URL configuration
client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

# 3. Make a request to the LLM
response = client.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{
      'type': 'function',
      'function': TOOL_FUNCTIONS["get_current_weather"]
    }],
)

# 4. Check if the LLM wants to use a tool
if 'tool_calls' in response['message']:
    tool_calls = response['message']['tool_calls']
    for tool_call in tool_calls:
        # 5. Execute the tool in your application
        if tool_call['function']['name'] == 'get_current_weather':
            city = tool_call['function']['arguments']['city']
            weather_info = get_weather(city)  # This happens in your code!
            
            # 6. Send the result back to the LLM
            final_response = client.chat(
                model=OLLAMA_CONFIG["default_model"],
                messages=[
                    {'role': 'user', 'content': 'What is the weather in Toronto?'},
                    response['message'],
                    {'role': 'tool', 'content': weather_info}  # Tool result
                ]
            )
            print(final_response['message']['content'])
```

## Multiple Tool Usage

Ollama supports scenarios where a model might need to call multiple tools, either in a single response or across a conversation. Here's how to implement multiple tool usage:

### Multiple Tools in a Single Call

Some models can return multiple tool calls in a single response. Your application needs to handle each tool call individually:

```python
from ollama import Client
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Implementation of your tools
def get_weather(city):
    return f"It's 22°C and sunny in {city}"

def get_population(city):
    populations = {
        "Toronto": "2.93 million",
        "New York": "8.8 million",
        "London": "8.9 million"
    }
    return populations.get(city, f"Population data for {city} not available")

client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

# Define multiple tools
response = client.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'What is the weather and population in Toronto?'}],
    tools=[
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_current_weather"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_population"]
        }
    ],
)

# Process all tool calls
if 'tool_calls' in response['message']:
    tool_calls = response['message']['tool_calls']
    
    # Store all the messages to be sent back
    conversation = [
        {'role': 'user', 'content': 'What is the weather and population in Toronto?'},
        response['message']
    ]
    
    # Process each tool call
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        args = tool_call['function']['arguments']
        
        if function_name == 'get_current_weather':
            result = get_weather(args['city'])
            conversation.append({'role': 'tool', 'content': result})
        
        elif function_name == 'get_population':
            result = get_population(args['city'])
            conversation.append({'role': 'tool', 'content': result})
    
    # Send all tool results back to the model
    final_response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=conversation
    )
    
    print(final_response['message']['content'])
```

### Sequential Tool Calls

In more complex scenarios, the model might need to make a sequence of tool calls based on previous results:

```python
from ollama import Client
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Tool implementations
def search_database(query):
    # Always return product results for widget-related queries
    if "widget" in query.lower() or "product" in query.lower():
        return "Found products: Widget A, Widget B, and Widget C"
    return "No results found for: " + query

def get_product_details(product_id):
    # Simulated product details lookup
    products = {
        "Widget A": {"price": "$10.99", "stock": 42, "category": "Tools"},
        "Widget B": {"price": "$24.99", "stock": 7, "category": "Electronics"},
        "Widget C": {"price": "$5.50", "stock": 0, "category": "Office Supplies"}
    }
    return str(products.get(product_id, "Product not found"))

# Initialize the client with base URL
client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

# Initial tools definition
tools = [
    {
        'type': 'function',
        'function': TOOL_FUNCTIONS["search_database"]
    },
    {
        'type': 'function',
        'function': TOOL_FUNCTIONS["get_product_details"]
    }
]

# Initial conversation
conversation = [
    {'role': 'user', 'content': 'I need details about widgets in stock'}
]

print("1. Sending initial request to Ollama...")
# First API call - model likely needs to search first
response = client.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=conversation,
    tools=tools
)

# Add the response to the conversation
conversation.append(response['message'])
print("2. Received initial response from Ollama")

# Process first tool call (likely search)
if 'tool_calls' in response['message']:
    print(f"3. Tool calls detected: {response['message']['tool_calls']}")
    tool_call = response['message']['tool_calls'][0]  # Get first tool call
    
    if tool_call['function']['name'] == 'search_database':
        query = tool_call['function']['arguments']['query']
        print(f"4. Executing search_database with query: {query}")
        search_result = search_database(query)
        print(f"5. Search result: {search_result}")
        
        # Add the tool result to the conversation
        conversation.append({'role': 'tool', 'content': search_result})
        
        print("6. Sending second request to Ollama...")
        # Second API call - model likely will ask for product details now
        response2 = client.chat(
            model=OLLAMA_CONFIG["default_model"],
            messages=conversation,
            tools=tools
        )
        
        # Add the second response to the conversation
        conversation.append(response2['message'])
        print("7. Received second response from Ollama")
        
        # Process second tool call (likely product details)
        if 'tool_calls' in response2['message']:
            print(f"8. Second tool calls detected: {response2['message']['tool_calls']}")
            tool_call2 = response2['message']['tool_calls'][0]
            
            if tool_call2['function']['name'] == 'get_product_details':
                product_id = tool_call2['function']['arguments']['product_id']
                print(f"9. Executing get_product_details with product_id: {product_id}")
                product_details = get_product_details(product_id)
                print(f"10. Product details: {product_details}")
                
                # Add the tool result to the conversation
                conversation.append({'role': 'tool', 'content': product_details})
                
                print("11. Sending final request to Ollama...")
                # Final response with all the information
                final_response = client.chat(
                    model=OLLAMA_CONFIG["default_model"],
                    messages=conversation,
                    tools=tools
                )
                
                print("\n=== Final Response ===")
                print(final_response['message']['content'])
            else:
                print(f"Error: Unexpected second tool call: {tool_call2['function']['name']}")
        else:
            print("Error: No tool calls found in second response")
            print(f"Second response content: {response2['message']['content']}")
    else:
        print(f"Error: Unexpected first tool call: {tool_call['function']['name']}")
else:
    print("Error: No tool calls found in initial response")
    print(f"Initial response content: {response['message']['content']}")
```

### Key Considerations for Multiple Tool Usage

1. **Tool Priority**: The LLM decides which tool to call based on the nature of the request
2. **Tool Dependencies**: Some tasks require sequential tool calls, where later calls depend on earlier results
3. **Conversation Management**: Keep track of the entire conversation, including all tool calls and responses
4. **Error Handling**: Implement robust error handling for cases where tool execution fails
5. **Parallel vs. Sequential**: Decide whether to execute multiple tool calls in parallel or in sequence

Multiple tool usage significantly expands the capabilities of LLMs, allowing them to accomplish complex tasks by breaking them down into a sequence of simpler operations, each handled by specialized tools.

## Hierarchical Agent Tool Usage

Ollama's tool function system can be extended to support hierarchical agents, where one agent can delegate tasks to other specialized agents. This enables complex task decomposition and parallel processing. Here's how to implement hierarchical agent tool usage:

### Agent Hierarchy Structure

```
┌─────────────────┐
│  Main Agent     │
│  (Coordinator)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sub-Agent 1    │
│  (Specialist)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sub-Agent 2    │
│  (Specialist)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Execution │
│  (Actual Work)  │
└─────────────────┘
```

### Implementation Example

```python
from ollama import Client
from typing import List, Dict, Any
from config import OLLAMA_CONFIG
import json
import logging

# Set up logging to capture all output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("hierarchical_agent_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)

# Initialize the client with base URL
client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

class Agent:
    def __init__(self, name: str, role: str, tools: List[Dict[str, Any]]):
        self.name = name
        self.role = role
        self.tools = tools
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})
        log_msg = f"[{self.name}] Added {role} message: {content[:50]}..." if len(content) > 50 else f"[{self.name}] Added {role} message: {content}"
        logging.info(log_msg)

    def chat(self, model: str = OLLAMA_CONFIG["default_model"]) -> Dict[str, Any]:
        logging.info(f"[{self.name}] Sending request to Ollama...")
        client = Client(
            host=OLLAMA_CONFIG["base_url"]
        )
        response = client.chat(
            model=model,
            messages=self.conversation,
            tools=self.tools
        )
        
        # Check if we got a proper response
        if 'message' in response and 'content' in response['message']:
            self.add_message("assistant", response['message']['content'])
            log_msg = f"[{self.name}] Received response from Ollama: {response['message']['content'][:50]}..." if len(response['message']['content']) > 50 else f"[{self.name}] Received response from Ollama: {response['message']['content']}"
            logging.info(log_msg)
        else:
            logging.info(f"[{self.name}] Warning: Received incomplete response from Ollama")
            
        return response

class CoordinatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Coordinator",
            role="Task coordinator and delegator",
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'delegate_task',
                        'description': 'Delegate a task to a specialized agent',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'task': {
                                    'type': 'string',
                                    'description': 'The task to delegate',
                                },
                                'agent': {
                                    'type': 'string', 
                                    'description': 'The specialized agent to handle the task',
                                    'enum': ['Researcher', 'Analyst']
                                },
                                'context': {
                                    'type': 'string',
                                    'description': 'Additional context for the task',
                                }
                            },
                            'required': ['task', 'agent', 'context'],
                        },
                    },
                }
            ]
        )

class ResearchAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Researcher",
            role="Research specialist",
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'search_database',
                        'description': 'Search for information in the database',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'query': {
                                    'type': 'string',
                                    'description': 'The search query',
                                },
                                'filters': {
                                    'type': 'object',
                                    'description': 'Search filters',
                                }
                            },
                            'required': ['query'],
                        },
                    },
                }
            ]
        )

class AnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Analyst",
            role="Data analysis specialist",
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'analyze_data',
                        'description': 'Analyze provided data',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'data': {
                                    'type': 'string',
                                    'description': 'The data to analyze',
                                },
                                'analysis_type': {
                                    'type': 'string',
                                    'description': 'Type of analysis to perform',
                                    'enum': ['trend', 'summary', 'prediction', 'comparison']
                                }
                            },
                            'required': ['data', 'analysis_type'],
                        },
                    },
                }
            ]
        )

def search_database(query, filters=None):
    """Actual implementation of the search database function"""
    logging.info(f"[Function] Executing search_database with query: {query}")
    if "electric" in query.lower() or "ev" in query.lower() or "vehicle" in query.lower():
        return """
        Search results for electric vehicles market trends:
        1. Global EV sales grew by 40% in 2022 compared to 2021
        2. Tesla maintains market leadership with 18% global market share
        3. China represents the largest EV market with over 50% of global sales
        4. European EV adoption increased by 15% year-over-year
        5. Battery costs have decreased by 35% over the past 5 years
        """
    return f"Search results for {query}: No specific data found. Please refine your search."

def analyze_data(data, analysis_type):
    """Actual implementation of the analyze data function"""
    logging.info(f"[Function] Executing analyze_data with type: {analysis_type}")
    if analysis_type == "trend":
        return """
        Trend Analysis of EV Market:
        - Consistent upward trajectory in global adoption rates
        - Accelerating growth in developing markets
        - Shift from sedans to SUV and crossover EV models
        - Increasing focus on battery technology and charging infrastructure
        - Growing competition with over 20 new EV models introduced annually
        """
    elif analysis_type == "summary":
        return """
        Summary of EV Market:
        - Total market value exceeds $500 billion globally
        - Major players include Tesla, BYD, Volkswagen Group, and GM
        - Battery electric vehicles dominate over plug-in hybrids
        - Government incentives remain a key driver of adoption
        """
    elif analysis_type == "prediction":
        return """
        Market Predictions for EV Sector:
        - Projected 25% annual growth rate through 2027
        - Battery price parity with ICE vehicles expected by 2025
        - Autonomous driving features to become standard in premium EV models
        - Solid-state battery technology breakthrough expected within 3 years
        """
    else:
        return f"Analysis of data using {analysis_type} method: Generic analysis results based on the provided data."

def execute_hierarchical_task(user_query: str):
    """Execute a task using hierarchical agents with proper error handling and debug output"""
    logging.info("\n=== Starting Hierarchical Task Execution ===")
    logging.info(f"User Query: {user_query}")
    
    # Initialize agents
    coordinator = CoordinatorAgent()
    researcher = ResearchAgent()
    analyst = AnalysisAgent()

    # Start with the coordinator 
    coordinator.add_message("user", f"""
Task: Research and analyze the market trends for electric vehicles in the last 5 years
Instructions:
1. First delegate the research task to the Researcher agent (only use 'Researcher' as the agent name)
2. The Researcher should search for information about electric vehicle market trends
3. Then delegate the analysis task to the Analyst agent (only use 'Analyst' as the agent name)
4. Finally, compile the results into a comprehensive report

Available agents: 'Researcher' and 'Analyst' only. Do not use any other agent names.
""")
    coordinator_response = coordinator.chat()
    
    # Default response in case the flow doesn't complete
    final_result = "No response generated. The agent workflow may not have completed properly."

    # Process coordinator's response 
    if 'message' not in coordinator_response or 'tool_calls' not in coordinator_response['message']:
        logging.info("[Error] Coordinator did not make any tool calls. Response content:")
        if 'message' in coordinator_response and 'content' in coordinator_response['message']:
            logging.info(coordinator_response['message']['content'])
        return coordinator_response['message']['content'] if 'message' in coordinator_response and 'content' in coordinator_response['message'] else final_result
    
    # Extract tool calls
    for tool_call in coordinator_response['message']['tool_calls']:
        if tool_call['function']['name'] == 'delegate_task':
            args = tool_call['function']['arguments']
            task = args.get('task', 'Unknown task')
            agent_name = args.get('agent', 'Unknown agent')
            context = args.get('context', 'No context provided')
            
            logging.info(f"[Coordinator] Delegating task '{task}' to {agent_name}")

            # Route task to appropriate agent
            if agent_name == "Researcher":
                researcher.add_message("user", f"Task: {task}\nContext: {context}")
                research_response = researcher.chat()
                
                if 'message' not in research_response or 'tool_calls' not in research_response['message']:
                    logging.info("[Error] Researcher did not make any tool calls")
                    coordinator.add_message("tool", "The researcher was unable to complete the task.")
                    final_response = coordinator.chat()
                    return final_response['message']['content'] if 'message' in final_response and 'content' in final_response['message'] else final_result
                
                for research_tool in research_response['message']['tool_calls']:
                    if research_tool['function']['name'] == 'search_database':
                        search_args = research_tool['function']['arguments']
                        query = search_args.get('query', 'default search')
                        filters = search_args.get('filters', None)
                        
                        # Execute search and get results
                        search_results = search_database(query, filters)
                        researcher.add_message("tool", search_results)
                        
                        # Get researcher's final thoughts
                        researcher_final = researcher.chat()
                        research_conclusion = researcher_final['message']['content']
                        
                        # Send results to the analyst with explicit instructions
                        analyst.add_message("user", f"""Task: Analyze the following research data on electric vehicle market trends.

Research data:
{search_results}

Researcher's conclusion:
{research_conclusion}

Instructions:
1. Please analyze this data using the 'analyze_data' function
2. Use analysis_type 'trend' to analyze trends in the electric vehicle market
3. Your response MUST call the analyze_data function with appropriate parameters
4. Do not provide analysis directly in your response - use the tool function instead
""")
                        analysis_response = analyst.chat()
                        
                        if 'message' not in analysis_response or 'tool_calls' not in analysis_response['message']:
                            logging.info("[Error] Analyst did not make any tool calls")
                            logging.info(f"Analyst response: {json.dumps(analysis_response, indent=2)}")
                            coordinator.add_message("tool", f"Research results: {search_results}\n\nThe analyst was unable to complete the analysis.")
                            final_response = coordinator.chat()
                            return final_response['message']['content'] if 'message' in final_response and 'content' in final_response['message'] else final_result
                        
                        for analysis_tool in analysis_response['message']['tool_calls']:
                            if analysis_tool['function']['name'] == 'analyze_data':
                                analysis_args = analysis_tool['function']['arguments']
                                data = analysis_args.get('data', search_results)
                                analysis_type = analysis_args.get('analysis_type', 'trend')
                                
                                # Execute analysis and get results
                                analysis_results = analyze_data(data, analysis_type)
                                analyst.add_message("tool", analysis_results)
                                
                                # Get analyst's final thoughts
                                analyst_final = analyst.chat()
                                analysis_conclusion = analyst_final['message']['content']
                                
                                # Send results back to coordinator
                                coordinator.add_message("tool", f"Research results: {search_results}\n\nAnalysis results: {analysis_results}\n\nAnalyst's conclusion: {analysis_conclusion}")
                                final_response = coordinator.chat()
                                final_result = final_response['message']['content']
                                
                                logging.info("\n=== Final Response ===")
                                logging.info(final_result)
                                return final_result
            
            elif agent_name == "Analyst":
                # Handle direct delegation to analyst if needed
                analyst.add_message("user", f"""Task: {task}
Context: {context}

Instructions:
1. You MUST use the analyze_data function to perform your analysis
2. Select an appropriate analysis_type from: 'trend', 'summary', 'prediction', or 'comparison'
3. Do not provide analysis directly - use the tool function
""")
                analysis_response = analyst.chat()
                
                if 'message' in analysis_response and 'content' in analysis_response['message']:
                    coordinator.add_message("tool", f"Analyst's response: {analysis_response['message']['content']}")
                    final_response = coordinator.chat()
                    final_result = final_response['message']['content']
                    
                    logging.info("\n=== Final Response ===")
                    logging.info(final_result)
                    return final_result
            
            else:
                logging.info(f"[Error] Unknown agent type: {agent_name}")
                coordinator.add_message("tool", f"Error: Agent {agent_name} not found.")
                final_response = coordinator.chat()
                return final_response['message']['content'] if 'message' in final_response and 'content' in final_response['message'] else final_result
        
        else:
            logging.info(f"[Error] Unknown tool call from coordinator: {tool_call['function']['name']}")
    
    logging.info("[Warning] No complete workflow was executed.")
    return final_result

if __name__ == "__main__":
    # Example usage
    result = execute_hierarchical_task("Research and analyze the market trends for electric vehicles in the last 5 years")
    print("\n=== Task Complete ===")
```

### Key Features of Hierarchical Agent Structure

1. **Agent Specialization**: Each agent has specific tools and capabilities
2. **Task Delegation**: The coordinator agent decides which specialists to assign tasks to
3. **Conversation Management**: Each agent maintains its own conversation history
4. **Error Handling**: Comprehensive checks to ensure the flow completes even if one agent fails
5. **Scalability**: The pattern can be extended to include more specialized agents or deeper hierarchies

Hierarchical agent tool usage allows for complex task processing by breaking tasks down and delegating to specialized agents. This approach is particularly useful for:

- Research and analysis tasks that require different expertise
- Complex workflows with multiple steps and dependencies
- Tasks that benefit from specialized knowledge or capabilities
- Scenarios where parallel processing of sub-tasks improves efficiency

## Real Executable Examples

This section provides real, executable examples using actual APIs and services. These examples can be run directly with the provided code.

### Example 1: Web Search and Analysis

This example uses DuckDuckGo for web search and OpenWeatherMap for weather data:

```python
from ollama import Client
import requests
from typing import Dict, Any
import json
from datetime import datetime
from config import OLLAMA_CONFIG, API_KEYS, TOOL_FUNCTIONS, API_ENDPOINTS, ERROR_MESSAGES

def get_weather(city: str) -> str:
    """Get weather data using OpenWeatherMap API"""
    if not API_KEYS["openweather"]:
        return ERROR_MESSAGES["missing_api_key"].format(service="OpenWeatherMap")
        
    url = API_ENDPOINTS["openweather"]
    params = {
        "q": city,
        "appid": API_KEYS["openweather"],
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code == 200:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        return f"Current weather in {city}: {description}, Temperature: {temp}°C, Humidity: {humidity}%"
    else:
        return f"Error getting weather for {city}: {data.get('message', 'Unknown error')}"

def search_web(query: str) -> str:
    """Search the web using DuckDuckGo API"""
    url = API_ENDPOINTS["duckduckgo"]
    params = {
        "q": query,
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract relevant information
    results = []
    if "Abstract" in data and data["Abstract"]:
        results.append(f"Summary: {data['Abstract']}")
    if "RelatedTopics" in data:
        for topic in data["RelatedTopics"][:3]:
            if "Text" in topic:
                results.append(topic["Text"])
    
    return "\n".join(results) if results else "No results found."

def analyze_text(text: str) -> str:
    """Analyze text using Ollama"""
    # Initialize the client with base URL
    client = Client(
        host=OLLAMA_CONFIG["base_url"]
    )
    
    response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=[{
            'role': 'user',
            'content': f"Analyze this text and provide key insights:\n\n{text}"
        }]
    )
    return response['message']['content']

def execute_research_task(query: str):
    """Execute a research task using multiple tools"""
    # Define tools
    tools = [
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["search_web"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_current_weather"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["analyze_text"]
        }
    ]

    # Initial conversation
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # Initialize the client with base URL
    client = Client(
        host=OLLAMA_CONFIG["base_url"]
    )
    
    # First API call
    response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=conversation,
        tools=tools
    )

    # Process tool calls
    if 'tool_calls' in response['message']:
        conversation.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if function_name == 'search_web':
                search_results = search_web(args['query'])
                conversation.append({'role': 'tool', 'content': search_results})
            
            elif function_name == 'get_current_weather':
                weather_info = get_weather(args['city'])
                conversation.append({'role': 'tool', 'content': weather_info})
            
            elif function_name == 'analyze_text':
                analysis = analyze_text(args['text'])
                conversation.append({'role': 'tool', 'content': analysis})

        # Get final response
        final_response = client.chat(
            model=OLLAMA_CONFIG["default_model"],
            messages=conversation
        )
        print(final_response['message']['content'])

# Example usage
if __name__ == "__main__":
    query = "What is the weather in Tokyo?"
    execute_research_task(query)
```

### Example 2: Financial Data Analysis

This example uses the Alpha Vantage API for financial data and Yahoo Finance for market data:

```python
from ollama import Client
import requests
import pandas as pd
from typing import Dict, Any
import yfinance as yf
from datetime import datetime, timedelta
from config import OLLAMA_CONFIG, API_KEYS, TOOL_FUNCTIONS, API_ENDPOINTS, ERROR_MESSAGES

client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

def get_stock_data(symbol: str) -> str:
    """Get stock data using Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        
        current_price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('volume', 'N/A')
        
        return f"""
Stock Data for {symbol}:
Current Price: ${current_price}
Market Cap: ${market_cap:,.2f}
Volume: {volume:,.0f}
1-Month High: ${hist['High'].max():.2f}
1-Month Low: ${hist['Low'].min():.2f}
"""
    except Exception as e:
        return f"Error getting stock data for {symbol}: {str(e)}"

def get_forex_data(from_symbol: str, to_symbol: str) -> str:
    """Get forex data using Alpha Vantage API"""
    if not API_KEYS["alpha_vantage"]:
        return ERROR_MESSAGES["missing_api_key"].format(service="Alpha Vantage")
        
    url = API_ENDPOINTS["alpha_vantage"]
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_symbol,
        "to_currency": to_symbol,
        "apikey": API_KEYS["alpha_vantage"]
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "Realtime Currency Exchange Rate" in data:
        rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return f"Current {from_symbol}/{to_symbol} exchange rate: {rate}"
    else:
        return f"Error getting forex data: {data.get('Note', 'Unknown error')}"

def analyze_financial_data(data: str) -> str:
    """Analyze financial data using Ollama"""
    # Initialize the client with base URL
    client = Client(
        host=OLLAMA_CONFIG["base_url"]
    )
    
    response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=[{
            'role': 'user',
            'content': f"Analyze this financial data and provide key insights:\n\n{data}"
        }]
    )
    return response['message']['content']

def execute_financial_analysis(query: str):
    """Execute a financial analysis task using multiple tools"""
    # Define tools
    tools = [
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_stock_data"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_forex_data"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["analyze_financial_data"]
        }
    ]

    # Initial conversation
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # Initialize the client with base URL
    client = Client(
        host=OLLAMA_CONFIG["base_url"]
    )
    
    # First API call
    response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=conversation,
        tools=tools
    )

    # Process tool calls
    if 'tool_calls' in response['message']:
        conversation.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if function_name == 'get_stock_data':
                stock_data = get_stock_data(args['symbol'])
                conversation.append({'role': 'tool', 'content': stock_data})
            
            elif function_name == 'get_forex_data':
                forex_data = get_forex_data(args['from_symbol'], args['to_symbol'])
                conversation.append({'role': 'tool', 'content': forex_data})
            
            elif function_name == 'analyze_financial_data':
                analysis = analyze_financial_data(args['data'])
                conversation.append({'role': 'tool', 'content': analysis})

        # Get final response
        final_response = client.chat(
            model=OLLAMA_CONFIG["default_model"],
            messages=conversation
        )
        print(final_response['message']['content'])

if __name__ == "__main__":
    query = "What is the latest price of Tesla?"
    execute_financial_analysis(query)
```

### Example 3: News and Sentiment Analysis

This example uses NewsAPI for news data and NLTK for sentiment analysis:

```python
from ollama import Client
import requests
from typing import Dict, Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from config import OLLAMA_CONFIG, API_KEYS, TOOL_FUNCTIONS, API_ENDPOINTS, ERROR_MESSAGES

client = Client(
    host=OLLAMA_CONFIG["base_url"]
)

# Download required NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_news(query: str) -> str:
    """Get news articles using NewsAPI"""
    if not API_KEYS["news_api"]:
        return ERROR_MESSAGES["missing_api_key"].format(service="NewsAPI")
        
    url = API_ENDPOINTS["news_api"]
    params = {
        "q": query,
        "apiKey": API_KEYS["news_api"],
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code == 200 and data["articles"]:
        articles = data["articles"][:5]  # Get top 5 articles
        results = []
        for article in articles:
            results.append(f"""
Title: {article['title']}
Source: {article['source']['name']}
Published: {article['publishedAt']}
URL: {article['url']}
""")
        return "\n".join(results)
    else:
        return f"Error getting news: {data.get('message', 'Unknown error')}"

def analyze_sentiment(text: str) -> str:
    """Analyze text sentiment using NLTK"""
    try:
        sentiment = sia.polarity_scores(text)
        return f"""
Sentiment Analysis Results:
Positive: {sentiment['pos']:.2f}
Negative: {sentiment['neg']:.2f}
Neutral: {sentiment['neu']:.2f}
Compound: {sentiment['compound']:.2f}
"""
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

def summarize_text(text: str) -> str:
    """Summarize text using Ollama"""
    # Initialize the client with base URL
    client = Client(
        host=OLLAMA_CONFIG["base_url"]
    )
    
    response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=[{
            'role': 'user',
            'content': f"Summarize this text concisely:\n\n{text}"
        }]
    )
    return response['message']['content']

def execute_news_analysis(query: str):
    """Execute a news analysis task using multiple tools"""
    # Define tools
    tools = [
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_news"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["analyze_sentiment"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["summarize_text"]
        }
    ]

    # Initial conversation
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # Initialize the client with base URL
    client = Client(
        host=OLLAMA_CONFIG["base_url"]
    )
    
    # First API call
    response = client.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=conversation,
        tools=tools
    )

    # Process tool calls
    if 'tool_calls' in response['message']:
        conversation.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if function_name == 'get_news':
                news_data = get_news(args['query'])
                conversation.append({'role': 'tool', 'content': news_data})
            
            elif function_name == 'analyze_sentiment':
                sentiment = analyze_sentiment(args['text'])
                conversation.append({'role': 'tool', 'content': sentiment})
            
            elif function_name == 'summarize_text':
                summary = summarize_text(args['text'])
                conversation.append({'role': 'tool', 'content': summary})

        # Get final response
        final_response = client.chat(
            model=OLLAMA_CONFIG["default_model"],
            messages=conversation
        )
        print(final_response['message']['content'])

if __name__ == "__main__":
    query = "What is the latest news on Bitcoin?"
    execute_news_analysis(query)
```

To use these examples:

1. Install required packages:
```bash
pip install ollama requests pandas yfinance nltk
```

2. Get API keys for the services:
   - OpenWeatherMap: https://openweathermap.org/api
   - Alpha Vantage: https://www.alphavantage.co/
   - NewsAPI: https://newsapi.org/

3. Replace the API key placeholders in the code with your actual API keys.

4. Run the examples:
```bash
python ollama_tool_functions.py
```

These examples demonstrate real-world usage of Ollama's tool function system with actual APIs and services. Each example shows different aspects of tool integration:

1. The web search example shows how to combine web search, weather data, and text analysis.
2. The financial analysis example demonstrates working with stock market and forex data.
3. The news analysis example shows how to integrate news retrieval with sentiment analysis and summarization.

Each example includes:
- Real API integrations
- Error handling
- Data processing
- Multiple tool usage
- Conversation management
- Result aggregation

The code is ready to run and can be extended with additional tools and capabilities as needed.

## Conclusion

Ollama's implementation of tool functions is a sophisticated system that combines:

1. **Flexible Templates**: Allowing for model-specific prompt formatting
2. **Intelligent Parsing**: Extracting structured data from model outputs
3. **Standardized API**: Providing consistent interfaces for developers

This architecture enables local LLMs to interact with external tools and systems, significantly expanding their capabilities beyond text generation. The implementation is designed to be:

- **Model-agnostic**: Works with different models that support tool calls
- **Format-flexible**: Adapts to various tool call representations
- **Developer-friendly**: Provides convenient APIs and OpenAI compatibility

As LLMs continue to evolve, Ollama's tool function implementation provides a solid foundation for building more capable AI systems that can seamlessly integrate with external tools and services. 
