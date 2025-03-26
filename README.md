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
import ollama

response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{
      'type': 'function',
      'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
          'type': 'object',
          'properties': {
            'city': {
              'type': 'string',
              'description': 'The name of the city',
            },
          },
          'required': ['city'],
        },
      },
    }],
)

print(response['message']['tool_calls'])
```

### OpenAI Compatibility Example

```python
import openai

openai.base_url = "http://localhost:11434/v1"
openai.api_key = 'ollama'

response = openai.chat.completions.create(
    model="llama3.1",
    messages=[{"role": "user", "content": "What's the weather in New York?"}],
    tools=[{
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "The name of the city",
            },
          },
          "required": ["city"],
        },
      },
    }],
)

print(response.choices[0].message.tool_calls)
```

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
import ollama

# 1. Define your tool implementation in your application
def get_weather(city):
    # Your actual API call or function implementation
    return f"It's 22°C and sunny in {city}"

# 2. Define the tool for the LLM
response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{
      'type': 'function',
      'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
          'type': 'object',
          'properties': {
            'city': {
              'type': 'string',
              'description': 'The name of the city',
            },
          },
          'required': ['city'],
        },
      },
    }],
)

# 3. Check if the LLM wants to use a tool
if 'tool_calls' in response['message']:
    tool_calls = response['message']['tool_calls']
    for tool_call in tool_calls:
        # 4. Execute the tool in your application
        if tool_call['function']['name'] == 'get_current_weather':
            city = tool_call['function']['arguments']['city']
            weather_info = get_weather(city)  # This happens in your code!
            
            # 5. Send the result back to the LLM
            final_response = ollama.chat(
                model='llama3.1',
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
import ollama

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

# Define multiple tools
response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'What is the weather and population in Toronto?'}],
    tools=[
        {
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'description': 'Get the current weather for a city',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'The name of the city',
                        },
                    },
                    'required': ['city'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_population',
                'description': 'Get the population for a city',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'The name of the city',
                        },
                    },
                    'required': ['city'],
                },
            },
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
    final_response = ollama.chat(
        model='llama3.1',
        messages=conversation
    )
    
    print(final_response['message']['content'])
```

### Sequential Tool Calls

In more complex scenarios, the model might need to make a sequence of tool calls based on previous results:

```python
import ollama

# Tool implementations
def search_database(query):
    # Simulated database search
    if "product" in query:
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

# Initial tools definition
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'search_database',
            'description': 'Search for products in the database',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query',
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_product_details',
            'description': 'Get details about a specific product',
            'parameters': {
                'type': 'object',
                'properties': {
                    'product_id': {
                        'type': 'string',
                        'description': 'The product identifier',
                    },
                },
                'required': ['product_id'],
            },
        },
    }
]

# Initial conversation
conversation = [
    {'role': 'user', 'content': 'I need details about widgets in stock'}
]

# First API call - model likely needs to search first
response = ollama.chat(
    model='llama3.1',
    messages=conversation,
    tools=tools
)

# Add the response to the conversation
conversation.append(response['message'])

# Process first tool call (likely search)
if 'tool_calls' in response['message']:
    tool_call = response['message']['tool_calls'][0]  # Get first tool call
    
    if tool_call['function']['name'] == 'search_database':
        query = tool_call['function']['arguments']['query']
        search_result = search_database(query)
        
        # Add the tool result to the conversation
        conversation.append({'role': 'tool', 'content': search_result})
        
        # Second API call - model likely will ask for product details now
        response2 = ollama.chat(
            model='llama3.1',
            messages=conversation,
            tools=tools
        )
        
        # Add the second response to the conversation
        conversation.append(response2['message'])
        
        # Process second tool call (likely product details)
        if 'tool_calls' in response2['message']:
            tool_call2 = response2['message']['tool_calls'][0]
            
            if tool_call2['function']['name'] == 'get_product_details':
                product_id = tool_call2['function']['arguments']['product_id']
                product_details = get_product_details(product_id)
                
                # Add the tool result to the conversation
                conversation.append({'role': 'tool', 'content': product_details})
                
                # Final response with all the information
                final_response = ollama.chat(
                    model='llama3.1',
                    messages=conversation,
                    tools=tools
                )
                
                print(final_response['message']['content'])
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
import ollama
from typing import List, Dict, Any

class Agent:
    def __init__(self, name: str, role: str, tools: List[Dict[str, Any]]):
        self.name = name
        self.role = role
        self.tools = tools
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def chat(self, model: str = 'llama3.1') -> Dict[str, Any]:
        response = ollama.chat(
            model=model,
            messages=self.conversation,
            tools=self.tools
        )
        self.add_message("assistant", response['message']['content'])
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
                                }
                            },
                            'required': ['data', 'analysis_type'],
                        },
                    },
                }
            ]
        )

def execute_hierarchical_task(user_query: str):
    # Initialize agents
    coordinator = CoordinatorAgent()
    researcher = ResearchAgent()
    analyst = AnalysisAgent()

    # Start with the coordinator
    coordinator.add_message("user", user_query)
    coordinator_response = coordinator.chat()

    # Process coordinator's response
    if 'tool_calls' in coordinator_response['message']:
        for tool_call in coordinator_response['message']['tool_calls']:
            if tool_call['function']['name'] == 'delegate_task':
                args = tool_call['function']['arguments']
                task = args['task']
                agent_name = args['agent']
                context = args['context']

                # Route task to appropriate agent
                if agent_name == "Researcher":
                    researcher.add_message("user", f"Task: {task}\nContext: {context}")
                    research_response = researcher.chat()
                    
                    if 'tool_calls' in research_response['message']:
                        for research_tool in research_response['message']['tool_calls']:
                            if research_tool['function']['name'] == 'search_database':
                                search_args = research_tool['function']['arguments']
                                # Execute search and get results
                                search_results = "Sample search results..."  # Replace with actual search
                                
                                # Delegate analysis to the analyst
                                analyst.add_message("user", f"Analyze this data: {search_results}")
                                analysis_response = analyst.chat()
                                
                                if 'tool_calls' in analysis_response['message']:
                                    for analysis_tool in analysis_response['message']['tool_calls']:
                                        if analysis_tool['function']['name'] == 'analyze_data':
                                            analysis_args = analysis_tool['function']['arguments']
                                            # Execute analysis and get results
                                            analysis_results = "Sample analysis results..."  # Replace with actual analysis
                                            
                                            # Send results back to coordinator
                                            coordinator.add_message("tool", f"Research and analysis results: {analysis_results}")
                                            final_response = coordinator.chat()
                                            print(final_response['message']['content'])

# Example usage
execute_hierarchical_task("Research and analyze the market trends for electric vehicles in the last 5 years")
```

### Key Features of Hierarchical Agent Implementation

1. **Agent Specialization**:
   - Each agent has specific tools and capabilities
   - Agents can be designed for different domains (research, analysis, decision-making, etc.)

2. **Task Delegation**:
   - Coordinator agent can delegate tasks to specialized agents
   - Tasks can be broken down into subtasks
   - Results can be aggregated and synthesized

3. **Conversation Management**:
   - Each agent maintains its own conversation history
   - Results are passed between agents through structured messages
   - Context is preserved throughout the task execution

4. **Error Handling and Recovery**:
   - Each agent can handle errors independently
   - Failed tasks can be retried or delegated to alternative agents
   - Results can be validated at each step

5. **Scalability**:
   - New agents can be added to handle different types of tasks
   - Agents can be organized in different hierarchies
   - Tasks can be processed in parallel or sequence

### Best Practices for Hierarchical Agent Implementation

1. **Clear Agent Roles**:
   - Define specific responsibilities for each agent
   - Use descriptive names and roles
   - Document agent capabilities and limitations

2. **Efficient Communication**:
   - Use structured message formats
   - Include relevant context in delegations
   - Maintain conversation history appropriately

3. **Task Decomposition**:
   - Break complex tasks into manageable subtasks
   - Consider dependencies between subtasks
   - Plan for result aggregation

4. **Error Handling**:
   - Implement robust error handling at each level
   - Provide fallback mechanisms
   - Log errors and recovery attempts

5. **Performance Optimization**:
   - Consider parallel processing where possible
   - Cache results when appropriate
   - Monitor and optimize agent interactions

Hierarchical agent tool usage enables complex task processing by leveraging multiple specialized agents, each with their own tools and capabilities. This approach is particularly useful for tasks that require different types of expertise or multiple processing steps.

## Real Executable Examples

This section provides real, executable examples using actual APIs and services. These examples can be run directly with the provided code.

### Example 1: Web Search and Analysis

This example uses DuckDuckGo for web search and OpenWeatherMap for weather data:

```python
import ollama
import requests
from typing import Dict, Any
import json
from datetime import datetime

# API Keys (replace with your own)
OPENWEATHER_API_KEY = "your_openweather_api_key"  # Get from https://openweathermap.org/api

def search_web(query: str) -> str:
    """Search the web using DuckDuckGo API"""
    url = "https://api.duckduckgo.com/"
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

def get_weather(city: str) -> str:
    """Get weather data using OpenWeatherMap API"""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
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

def analyze_text(text: str) -> str:
    """Analyze text using Ollama"""
    response = ollama.chat(
        model='llama3.1',
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
            'function': {
                'name': 'search_web',
                'description': 'Search the web for information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query',
                        }
                    },
                    'required': ['query'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get current weather for a city',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'The name of the city',
                        }
                    },
                    'required': ['city'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'analyze_text',
                'description': 'Analyze text and provide insights',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'The text to analyze',
                        }
                    },
                    'required': ['text'],
                },
            },
        }
    ]

    # Initial conversation
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # First API call
    response = ollama.chat(
        model='llama3.1',
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
            
            elif function_name == 'get_weather':
                weather_info = get_weather(args['city'])
                conversation.append({'role': 'tool', 'content': weather_info})
            
            elif function_name == 'analyze_text':
                analysis = analyze_text(args['text'])
                conversation.append({'role': 'tool', 'content': analysis})

        # Get final response
        final_response = ollama.chat(
            model='llama3.1',
            messages=conversation
        )
        print(final_response['message']['content'])

# Example usage
if __name__ == "__main__":
    # Replace with your OpenWeatherMap API key
    OPENWEATHER_API_KEY = "your_openweather_api_key"
    
    # Example queries
    queries = [
        "What's the weather in Tokyo and what are the latest developments in quantum computing?",
        "Research the impact of climate change on polar bears and analyze the findings",
        "What's the current state of AI regulation in the EU and how does it affect startups?"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        execute_research_task(query)
```

### Example 2: Financial Data Analysis

This example uses the Alpha Vantage API for financial data and Yahoo Finance for market data:

```python
import ollama
import requests
import pandas as pd
from typing import Dict, Any
import yfinance as yf
from datetime import datetime, timedelta

# API Keys (replace with your own)
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"  # Get from https://www.alphavantage.co/

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
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_symbol,
        "to_currency": to_symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
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
    response = ollama.chat(
        model='llama3.1',
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
            'function': {
                'name': 'get_stock_data',
                'description': 'Get stock market data for a symbol',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'symbol': {
                            'type': 'string',
                            'description': 'The stock symbol (e.g., AAPL, GOOGL)',
                        }
                    },
                    'required': ['symbol'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_forex_data',
                'description': 'Get forex exchange rate data',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'from_symbol': {
                            'type': 'string',
                            'description': 'The source currency symbol (e.g., USD, EUR)',
                        },
                        'to_symbol': {
                            'type': 'string',
                            'description': 'The target currency symbol (e.g., USD, EUR)',
                        }
                    },
                    'required': ['from_symbol', 'to_symbol'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'analyze_financial_data',
                'description': 'Analyze financial data and provide insights',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'data': {
                            'type': 'string',
                            'description': 'The financial data to analyze',
                        }
                    },
                    'required': ['data'],
                },
            },
        }
    ]

    # Initial conversation
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # First API call
    response = ollama.chat(
        model='llama3.1',
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
        final_response = ollama.chat(
            model='llama3.1',
            messages=conversation
        )
        print(final_response['message']['content'])

# Example usage
if __name__ == "__main__":
    # Replace with your Alpha Vantage API key
    ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"
    
    # Example queries
    queries = [
        "Analyze the performance of AAPL and GOOGL stocks and compare them",
        "What's the current EUR/USD exchange rate and how does it affect European exports?",
        "Research the impact of recent market trends on tech stocks and provide insights"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        execute_financial_analysis(query)
```

### Example 3: News and Sentiment Analysis

This example uses NewsAPI for news data and NLTK for sentiment analysis:

```python
import ollama
import requests
from typing import Dict, Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# API Keys (replace with your own)
NEWS_API_KEY = "your_news_api_key"  # Get from https://newsapi.org/

# Download required NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_news(query: str) -> str:
    """Get news articles using NewsAPI"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
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
    response = ollama.chat(
        model='llama3.1',
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
            'function': {
                'name': 'get_news',
                'description': 'Get recent news articles about a topic',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query for news',
                        }
                    },
                    'required': ['query'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'analyze_sentiment',
                'description': 'Analyze the sentiment of text',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'The text to analyze',
                        }
                    },
                    'required': ['text'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'summarize_text',
                'description': 'Summarize text concisely',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'The text to summarize',
                        }
                    },
                    'required': ['text'],
                },
            },
        }
    ]

    # Initial conversation
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # First API call
    response = ollama.chat(
        model='llama3.1',
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
        final_response = ollama.chat(
            model='llama3.1',
            messages=conversation
        )
        print(final_response['message']['content'])

# Example usage
if __name__ == "__main__":
    # Replace with your NewsAPI key
    NEWS_API_KEY = "your_news_api_key"
    
    # Example queries
    queries = [
        "Analyze the sentiment of recent news about artificial intelligence",
        "Summarize the latest developments in renewable energy and analyze public sentiment",
        "Research recent news about climate change and provide a comprehensive analysis"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
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
