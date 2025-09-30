#!/usr/bin/env python3
"""
OpenShift etcd Analyzer MCP Client with FastAPI and LangGraph Integration
Provides AI-powered chat interface for interacting with etcd analysis MCP tools
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, AsyncGenerator
import aiohttp
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, AsyncGenerator

# MCP and LangGraph imports
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# LangChain and LangGraph imports
from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from elt.etcd_analyzer_elt_json2table import json_to_html_table


import warnings
# Suppress urllib3 deprecation warning triggered by kubernetes client
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"HTTPResponse\.getheaders\(\) is deprecated"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _SuppressInvalidHttp(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Invalid HTTP request received" not in msg

# Suppress noisy warnings from various Uvicorn HTTP implementations
logging.getLogger("uvicorn").addFilter(_SuppressInvalidHttp())
logging.getLogger("uvicorn.error").addFilter(_SuppressInvalidHttp())
logging.getLogger("uvicorn.access").addFilter(_SuppressInvalidHttp())
logging.getLogger("uvicorn.protocols.http.h11_impl").addFilter(_SuppressInvalidHttp())
logging.getLogger("uvicorn.protocols.http.httptools_impl").addFilter(_SuppressInvalidHttp())

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message to process")
    conversation_id: str = Field(default="default", description="Conversation identifier")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt override")

class SystemPromptRequest(BaseModel):
    """System prompt configuration request"""
    system_prompt: str = Field(..., description="System prompt to set")
    conversation_id: str = Field(default="default", description="Conversation identifier")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Check timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")

class MCPTool(BaseTool):
    """LangChain tool wrapper for MCP tools"""
    
    name: str
    description: str
    mcp_client: 'MCPClient'
    
    def __init__(self, name: str, description: str, mcp_client: 'MCPClient'):
        # Initialize BaseTool (pydantic model) with fields
        super().__init__(name=name, description=description, mcp_client=mcp_client)
    
    def _run(self, **kwargs) -> str:
        """Synchronous run - not used"""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous tool execution"""
        try:
            # Unwrap nested 'kwargs' that some agents pass, and drop None values
            params = kwargs.get('kwargs', kwargs) or {}
            if isinstance(params, dict):
                params = {k: v for k, v in params.items() if v is not None}
            result = await self.mcp_client.call_tool(self.name, params)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.name}: {e}")
            return f"Error: {str(e)}"

class MCPClient:
    """MCP Client for interacting with the etcd analyzer server"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.mcp_server_url = mcp_server_url
        self.session = None
        self.available_tools: List[Dict[str, Any]] = []
        self.langchain_tools: List[MCPTool] = []
    
    async def connect(self):
        """Connect to MCP server and initialize tools"""
        try:
            url = f"{self.mcp_server_url}/mcp"
            
            # Connect to the server using Streamable HTTP
            async with streamablehttp_client(url) as (
                read_stream,
                write_stream,
                get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the connection
                    await session.initialize()
 
                    # Get session id once connection established
                    session_id = get_session_id()
                    logger.info(f"Session ID: {session_id}")
                    tools_result = await session.list_tools()
                    logger.info(f"Discovered {len(tools_result.tools)} MCP tools")
            
                    for tool in tools_result.tools:
                        logger.info(f"Tool discovered: {tool.name}")
                        # Handle schema that may be a dict or a Pydantic model
                        input_schema = {}
                        try:
                            if tool.inputSchema:
                                if isinstance(tool.inputSchema, dict):
                                    input_schema = tool.inputSchema
                                elif hasattr(tool.inputSchema, "model_dump"):
                                    input_schema = tool.inputSchema.model_dump()
                                elif hasattr(tool.inputSchema, "dict"):
                                    input_schema = tool.inputSchema.dict()
                        except Exception:
                            input_schema = {}
                        
                        self.available_tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": input_schema
                        })
                    
                    self._create_langchain_tools()
                    logger.info(f"Loaded {len(self.available_tools)} MCP tools")
            return True
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    def _create_langchain_tools(self):
        """Create LangChain tool wrappers for MCP tools"""
        self.langchain_tools = []
        for tool_info in self.available_tools:
            langchain_tool = MCPTool(
                name=tool_info["name"],
                description=tool_info["description"],
                mcp_client=self
            )
            self.langchain_tools.append(langchain_tool)
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call an MCP tool with parameters"""
        try:
            url = f"{self.mcp_server_url}/mcp"
            async with streamablehttp_client(url) as (
                read_stream,
                write_stream,
                get_session_id,
            ):
                try:
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        if get_session_id:
                            logger.info(f"Session ID in call_tool: {get_session_id()}")
                        logger.info(f"Calling tool {tool_name} with params {params}")

                        # Pass parameters directly according to the tool's input schema
                        result = await session.call_tool(tool_name, params or {})
                        # logger.info(f"Tool {tool_name} result is: {result}")

                        if not result.content:
                            logger.warning(f"Tool {tool_name} returned empty content")
                            return {"error": "Empty response from tool", "tool": tool_name}

                        if len(result.content) == 0:
                            logger.warning(f"Tool {tool_name} returned no content items")
                            return {"error": "No content items in response", "tool": tool_name}

                        content_text = result.content[0].text
                        if not content_text or content_text.strip() == "":
                            logger.warning(f"Tool {tool_name} returned empty text content")
                            return {"error": "Empty text content from tool", "tool": tool_name}

                        # Handle different response formats
                        content_text = content_text.strip()

                        # If content is empty after stripping
                        if not content_text:
                            return {"error": "Empty response after trimming whitespace", "tool": tool_name}

                        # Try to parse as JSON first only if it looks like JSON
                        try:
                            if content_text[:1] in ['{', '[']:
                                
                                json_data = json.loads(content_text)
                                logger.info(f"json_data of {tool_name} in call_tool is: {json_data}")
                                # Format etcd analyzer results appropriately
                                if tool_name in ["get_server_health"]:
                                    return json_data
                                else:
                                    # Format as HTML table for structured display
                                    return json_to_html_table(json_data)
                                    # return json_data
                            else:
                                # Plain text/HTML response; return as-is string so UI can render directly
                                return content_text

                        except json.JSONDecodeError as json_err:
                            logger.error(
                                f"Failed to parse JSON from tool {tool_name}. Content: '{content_text[:200]}...'"
                            )
                            logger.error(f"JSON decode error: {json_err}")

                            # Check if it's a simple string response that should be JSON
                            if content_text.startswith('{') or content_text.startswith('['):
                                # Looks like malformed JSON
                                return {
                                    "error": f"Malformed JSON response: {str(json_err)}",
                                    "tool": tool_name,
                                    "raw_content": content_text[:500],
                                    "content_type": "malformed_json",
                                }
                            else:
                                # Return as plain text response
                                return {
                                    "result": content_text,
                                    "tool": tool_name,
                                    "content_type": "text",
                                    "message": "Tool returned plain text instead of JSON",
                                }
                except Exception as e:
                    logger.error(f"MCP ClientSession error for tool {tool_name}: {e}")
                    # Return structured error instead of raising to avoid ExceptionGroup on __aexit__
                    return {
                        "error": str(e),
                        "tool": tool_name,
                        "error_type": type(e).__name__,
                    }
        except Exception as outer_e:
            logger.error(f"Error calling tool {tool_name}: {outer_e}")
            return {
                "error": str(outer_e),
                "tool": tool_name,
                "error_type": type(outer_e).__name__,
            }
            
    async def check_mcp_connectivity_health(self):
        """Check MCP server connectivity health"""
        try:
            url = f"{self.mcp_server_url}/mcp"

            async with streamablehttp_client(url) as (
                    read_stream,
                    write_stream,
                    get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
 
                    session_id = get_session_id()
                    logger.info(f"Health check session ID: {session_id}")
                    if session_id:
                       return {
                        "status": "healthy",
                        "mcp_connection": "ok",
                        "last_check": datetime.now(timezone.utc).isoformat()
                      }
                    else:
                       return {
                        "status": "unhealthy",
                        "mcp_connection": "disconnected",
                        "last_check": datetime.now(timezone.utc).isoformat()
                      }
                    
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to connect to MCP server: {e}")
        return {
            "status": "unhealthy",
            "mcp_connection": "disconnected", 
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }

    async def check_etcd_analyzer_health(self) -> Dict[str, Any]:
        """Check etcd analyzer server health via MCP tools"""
        try:
            # Test basic connectivity first
            await self.connect()
            
            # Try the server health check tool
            health_tools = [
                "get_server_health"
            ]
            
            health_result = None
            tool_used = None
            
            for tool_name in health_tools:
                # Check if tool exists in available tools
                if any(tool["name"] == tool_name for tool in self.available_tools):
                    try:
                        logger.info(f"Trying health check with tool: {tool_name}")
                        health_result = await self.call_tool(tool_name, {})
                        
                        # Check if we got a valid result
                        if (isinstance(health_result, dict) and 
                            "error" not in health_result and 
                            health_result):
                            tool_used = tool_name
                            break
                        elif isinstance(health_result, dict) and "error" in health_result:
                            logger.warning(f"Tool {tool_name} returned error: {health_result.get('error')}")
                            continue
                            
                    except Exception as tool_error:
                        logger.warning(f"Tool {tool_name} failed: {str(tool_error)}")
                        continue
            
            # Analyze the health result
            if health_result and tool_used:
                overall_health = health_result.get("status", "unknown")
                collectors_initialized = health_result.get("collectors_initialized", False)
                
                return {
                    "status": "healthy" if collectors_initialized else "partial",
                    "etcd_analyzer_connection": "ok",
                    "tools_available": len(self.available_tools),
                    "collectors_initialized": collectors_initialized,
                    "overall_health": overall_health,
                    "tool_used": tool_used,
                    "last_check": datetime.now(timezone.utc).isoformat(),
                    "health_details": health_result
                }
            else:
                # No tools worked, but MCP connection is established
                return {
                    "status": "partial",
                    "etcd_analyzer_connection": "error",
                    "tools_available": len(self.available_tools),
                    "tool_error": "Health check tools failed or returned empty responses",
                    "overall_health": "unknown",
                    "last_check": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"etcd analyzer connectivity check failed: {e}")
            return {
                "status": "unhealthy",
                "etcd_analyzer_connection": "unknown", 
                "error": str(e),
                "overall_health": "unknown",
                "last_check": datetime.now(timezone.utc).isoformat()
            }

class ChatBot:
    """LangGraph-powered chatbot for etcd analyzer MCP interaction"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.memory = MemorySaver()
        self.conversations: Dict[str, CompiledStateGraph] = {}
        self.conversation_system_prompts: Dict[str, str] = {}  # Store system prompts per conversation
        
        # Default system prompt for etcd analysis
        self.default_system_prompt = """You are an expert OpenShift etcd performance analyst with deep knowledge of:
- etcd cluster health monitoring and diagnostics
- OpenShift cluster troubleshooting and performance optimization
- Distributed systems analysis and debugging
- Performance metrics interpretation and correlation

When analyzing etcd data:
1. Focus on performance bottlenecks, health issues, and optimization opportunities
2. Provide actionable insights and specific recommendations
3. Explain technical findings in clear, structured responses
4. Use the available MCP tools to gather comprehensive analysis
5. Correlate multiple metrics to identify root causes
6. Prioritize critical issues that affect cluster stability

Always structure your responses with:
- Executive summary of findings
- Detailed technical analysis
- Specific recommendations with priority levels
- Next steps for investigation or remediation

Be thorough but concise, and always explain the business impact of technical issues."""

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")        

        self.llm = ChatOpenAI(
            model="gemini-2.5-pro",
            base_url=base_url,
            api_key=api_key,
            temperature=0.1,
            streaming=True
        )

    def set_system_prompt(self, conversation_id: str, system_prompt: str):
        """Set system prompt for a specific conversation"""
        self.conversation_system_prompts[conversation_id] = system_prompt
        # If conversation already exists, remove it to recreate with new system prompt
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def get_system_prompt(self, conversation_id: str) -> str:
        """Get system prompt for a conversation"""
        return self.conversation_system_prompts.get(conversation_id, self.default_system_prompt)

    def get_or_create_agent(self, conversation_id: str) -> CompiledStateGraph:
        """Get or create a conversation agent with memory and system prompt"""
        if conversation_id not in self.conversations:
            # Get system prompt for this conversation
            system_prompt = self.get_system_prompt(conversation_id)
            
            # Create agent with system prompt and memory
            agent = create_react_agent(
                self.llm,
                self.mcp_client.langchain_tools,
                checkpointer=self.memory
            )
            self.conversations[conversation_id] = agent
            
        return self.conversations[conversation_id]
    
    async def chat_stream(self, message: str, conversation_id: str, system_prompt: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response with proper message streaming that preserves tool results"""
        try:
            # Set system prompt if provided
            if system_prompt:
                self.set_system_prompt(conversation_id, system_prompt)
            
            agent = self.get_or_create_agent(conversation_id)
            
            # Create thread config for this conversation
            config = {"configurable": {"thread_id": conversation_id}}
            
            tool_results_sent = []  # Track tool results that have been sent
            last_ai_content = ""    # Track AI content for proper streaming
            
            # Stream the response
            # Always include system prompt as a system message to ensure instructions are applied
            system_prompt_effective = self.get_system_prompt(conversation_id)
            input_messages = [("system", system_prompt_effective), ("user", message)]

            async for chunk in agent.astream(
                {"messages": input_messages},
                config=config
            ):
                # Process different types of chunks
                if "agent" in chunk:
                    agent_data = chunk["agent"]
                    if "messages" in agent_data:
                        for msg in agent_data["messages"]:
                            if hasattr(msg, 'content') and msg.content:
                                # Stream AI content incrementally
                                content = str(msg.content)
                                
                                # For AI messages, send them as streaming updates
                                if content != last_ai_content:
                                    last_ai_content = content
                                    yield {
                                        "type": "message",
                                        "content": content,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "streaming": True,
                                        "tool_result": False
                                    }
                
                elif "tools" in chunk:
                    tool_data = chunk["tools"]
                    if "messages" in tool_data:
                        for msg in tool_data["messages"]:
                            if hasattr(msg, 'content') and msg.content:
                                # Format tool result for display
                                try:
                                    content = msg.content
                                    
                                    # Process different content types
                                    if isinstance(content, dict):
                                        # For HTML tables, pass through directly
                                        if any(key in content for key in ['html', 'table', 'content']):
                                            formatted_content = content.get('html') or content.get('table') or content.get('content') or json.dumps(content, indent=2)
                                        else:
                                            formatted_content = json.dumps(content, indent=2)
                                    elif isinstance(content, str):
                                        # Check if it's HTML table content
                                        if content.strip().startswith('<table') or content.strip().startswith('<!DOCTYPE') or '<html>' in content:
                                            formatted_content = content
                                        elif content[:1] in ['{', '[']:
                                            try:
                                                parsed = json.loads(content)
                                                formatted_content = json.dumps(parsed, indent=2)
                                            except:
                                                formatted_content = content
                                        else:
                                            formatted_content = content
                                    else:
                                        formatted_content = str(content)
                                    
                                    # Only send tool results once
                                    tool_result_id = hash(str(content))
                                    if tool_result_id not in tool_results_sent:
                                        tool_results_sent.append(tool_result_id)
                                        
                                        yield {
                                            "type": "message",
                                            "content": formatted_content,
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "tool_result": True,
                                            "streaming": False
                                        }
                                except Exception as e:
                                    logger.error(f"Error formatting tool result: {e}")
                                    yield {
                                        "type": "message",
                                        "content": str(msg.content),
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "tool_result": True,
                                        "streaming": False
                                    }
            
            # Send final completion signal
            yield {
                "type": "message_complete",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
                            
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield {
                "type": "error",
                "content": f"An error occurred: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def chat_stream(self, message: str, conversation_id: str, system_prompt: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response with proper message streaming"""
        try:
            # Set system prompt if provided
            if system_prompt:
                self.set_system_prompt(conversation_id, system_prompt)
            
            agent = self.get_or_create_agent(conversation_id)
            
            # Create thread config for this conversation
            config = {"configurable": {"thread_id": conversation_id}}
            
            accumulated_content = ""
            
            # Stream the response
            # Always include system prompt as a system message to ensure instructions are applied
            system_prompt_effective = self.get_system_prompt(conversation_id)
            input_messages = [("system", system_prompt_effective), ("user", message)]

            async for chunk in agent.astream(
                {"messages": input_messages},
                config=config
            ):
                # Process different types of chunks
                if "agent" in chunk:
                    agent_data = chunk["agent"]
                    if "messages" in agent_data:
                        for msg in agent_data["messages"]:
                            if hasattr(msg, 'content') and msg.content:
                                # Stream content incrementally
                                content = str(msg.content)
                                
                                # For streaming, we want to send incremental updates
                                if content != accumulated_content:
                                    accumulated_content = content
                                    yield {
                                        "type": "message",
                                        "content": content,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "streaming": True
                                    }
                
                elif "tools" in chunk:
                    tool_data = chunk["tools"]
                    if "messages" in tool_data:
                        for msg in tool_data["messages"]:
                            if hasattr(msg, 'content') and msg.content:
                                # Format tool result for display
                                try:
                                    content = msg.content
                                    # If dict → pretty JSON; if JSON string → parse then pretty; else pass through as-is
                                    if isinstance(content, dict):
                                        # For HTML tables, pass through directly
                                        if any(key in content for key in ['html', 'table', 'content']):
                                            out = content.get('html') or content.get('table') or content.get('content') or json.dumps(content, indent=2)
                                        else:
                                            out = json.dumps(content, indent=2)
                                    elif isinstance(content, str):
                                        # Check if it's HTML table content
                                        if content.strip().startswith('<table') or content.strip().startswith('<!DOCTYPE') or '<html>' in content:
                                            out = content
                                        elif content[:1] in ['{', '[']:
                                            try:
                                                parsed = json.loads(content)
                                                out = json.dumps(parsed, indent=2)
                                            except:
                                                out = content
                                        else:
                                            out = content
                                    else:
                                        out = str(content)
                                    
                                    yield {
                                        "type": "message", 
                                        "content": f"\n\n{out}",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "tool_result": True
                                    }
                                except Exception:
                                    yield {
                                        "type": "message",
                                        "content": f"\n\n{str(msg.content)}",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "tool_result": True
                                    }
            
            # Send final completion signal
            yield {
                "type": "message_complete",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
                            
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield {
                "type": "error",
                "content": f"An error occurred: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global instances
mcp_client = MCPClient()
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    global chatbot
    
    logger.info("Starting etcd Analyzer MCP Client...")
    
    # Initialize MCP client
    connected = await mcp_client.connect()
    if connected:
        logger.info("etcd Analyzer MCP Client connected successfully")
        # Initialize chatbot
        chatbot = ChatBot(mcp_client)
    else:
        logger.warning("Failed to connect to MCP server, continuing with limited functionality")
        chatbot = ChatBot(mcp_client)  # Create anyway for graceful degradation
    
    yield
    
    logger.info("Shutting down etcd Analyzer MCP Client...")

# Create FastAPI app
app = FastAPI(
    title="etcd Analyzer MCP Client",
    description="AI-powered chat interface for OpenShift etcd performance analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI HTML
HTML_FILE_PATH = os.path.join(os.path.dirname(__file__), "webroot", "etcd_analyzer_mcp_llm.html")

@app.get("/", include_in_schema=False)
async def serve_root_html():
    if os.path.exists(HTML_FILE_PATH):
        return FileResponse(HTML_FILE_PATH, media_type="text/html")
    raise HTTPException(status_code=404, detail="etcd_analyzer_mcp_llm.html not found")

@app.get("/ui", include_in_schema=False)
async def serve_ui_html():
    if os.path.exists(HTML_FILE_PATH):
        return FileResponse(HTML_FILE_PATH, media_type="text/html")
    raise HTTPException(status_code=404, detail="etcd_analyzer_mcp_llm.html not found")

# Health endpoints
@app.get("/api/mcp/health", response_model=HealthResponse)
async def mcp_health_check():
    """MCP connectivity health check endpoint"""
    health_data = await mcp_client.check_mcp_connectivity_health()
    return HealthResponse(
        status=health_data["status"],
        timestamp=health_data["last_check"],
        details=health_data
    )

@app.get("/api/etcd/health", response_model=HealthResponse)
async def etcd_health_check():
    """etcd analyzer health check endpoint"""
    health_data = await mcp_client.check_etcd_analyzer_health()
    return HealthResponse(
        status=health_data["status"],
        timestamp=health_data["last_check"],
        details=health_data
    )

@app.get("/api/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "tools": mcp_client.available_tools,
        "count": len(mcp_client.available_tools)
    }

@app.post("/api/tools/{tool_name}")
async def call_tool_direct(tool_name: str, params: Dict[str, Any] = None):
    """Direct tool call endpoint"""
    try:
        result = await mcp_client.call_tool(tool_name, params or {})
        logger.info(f"Direct tool call result type: {type(result)}")
        
        # If the tool already returned a formatted string, return as is
        if isinstance(result, str):
            return result
        # If the result is a dict with an error from the tool call, pass through
        if isinstance(result, dict) and result.get("error") and result.get("tool"):
            return result
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system-prompt")
async def set_system_prompt(request: SystemPromptRequest):
    """Set system prompt for a conversation"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        chatbot.set_system_prompt(request.conversation_id, request.system_prompt)
        return {
            "status": "success",
            "message": f"System prompt updated for conversation {request.conversation_id}",
            "conversation_id": request.conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set system prompt: {str(e)}")

@app.get("/api/system-prompt/{conversation_id}")
async def get_system_prompt(conversation_id: str):
    """Get current system prompt for a conversation"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        system_prompt = chatbot.get_system_prompt(conversation_id)
        return {
            "conversation_id": conversation_id,
            "system_prompt": system_prompt,
            "is_default": system_prompt == chatbot.default_system_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system prompt: {str(e)}")

@app.delete("/api/system-prompt/{conversation_id}")
async def reset_system_prompt(conversation_id: str):
    """Reset system prompt to default for a conversation"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        if conversation_id in chatbot.conversation_system_prompts:
            del chatbot.conversation_system_prompts[conversation_id]
        # Remove existing conversation to recreate with default prompt
        if conversation_id in chatbot.conversations:
            del chatbot.conversations[conversation_id]
        
        return {
            "status": "success",
            "message": f"System prompt reset to default for conversation {conversation_id}",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset system prompt: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with proper message streaming"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    async def generate():
        try:
            async for chunk in chatbot.chat_stream(request.message, request.conversation_id, request.system_prompt):
                # Format as Server-Sent Events
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_chunk = {
                "type": "error",
                "content": f"Stream error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # Send final completion signal
        completion_chunk = {
            "type": "stream_end",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        yield f"data: {json.dumps(completion_chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.post("/chat")
async def chat_simple(request: ChatRequest):
    """Simple chat endpoint (non-streaming)"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    responses = []
    async for chunk in chatbot.chat_stream(request.message, request.conversation_id, request.system_prompt):
        responses.append(chunk)
    
    return {"responses": responses}

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "etcd_analyzer_client_chat:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )