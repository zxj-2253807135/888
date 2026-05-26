"""
LangServe 后端服务 - 支持高并发访问
集成 LangSmith 监控
"""

import os
import sys
from typing import List, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量（必须在导入LangChain模块之前）
load_dotenv()

# 配置LangSmith（必须在创建任何LangChain对象之前）
if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_API_KEY") != "your_langsmith_api_key_here":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "langchain-chat-assistant")

from langserve import add_routes
from langchain_core.messages import HumanMessage, AIMessage

from app.chains.chat_chain import ChatAssistant, create_simple_chain, create_agent_chain
from app.chains.local_chain import LocalChatAssistant


# ============ Pydantic 模型定义 ============

class ChatRequest(BaseModel):
    """聊天请求模型"""
    input: str = Field(description="用户输入的消息")
    chat_history: List[Dict[str, str]] = Field(default=[], description="对话历史")
    session_id: str = Field(default="default", description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    output: str = Field(description="助手回复")
    chat_history: List[Dict[str, str]] = Field(description="更新后的对话历史")


class StreamRequest(BaseModel):
    """流式请求模型"""
    input: str = Field(description="用户输入的消息")
    session_id: str = Field(default="default", description="会话ID")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    langsmith_enabled: bool
    api_configured: bool


# ============ 会话管理 ============

# 简单的内存会话存储（生产环境建议使用Redis）
sessions: Dict[str, ChatAssistant] = {}


def get_or_create_session(session_id: str) -> ChatAssistant:
    """获取或创建会话（使用DeepSeek API）"""
    if session_id not in sessions:
        sessions[session_id] = ChatAssistant(use_tools=True)
    return sessions[session_id]


def clear_session(session_id: str):
    """清除会话"""
    if session_id in sessions:
        del sessions[session_id]


# ============ FastAPI 应用 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("[START] LangChain Chat Assistant Server 启动中...")
    print(f"[INFO] LangSmith 监控: {'已启用' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else '未启用'}")
    yield
    # 关闭时执行
    print("[EXIT] 服务器关闭")


app = FastAPI(
    title="LangChain Chat Assistant API",
    description="基于LangChain的智能对话助手 - 集成LangSmith监控",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ API 路由 ============

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "LangChain Chat Assistant API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_configured = api_key is not None and api_key != "your_deepseek_api_key_here"
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        langsmith_enabled=os.getenv("LANGCHAIN_TRACING_V2") == "true",
        api_configured=api_configured
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    对话端点 - 使用DeepSeek API，支持工具调用和记忆功能
    """
    try:
        # 获取或创建会话
        assistant = get_or_create_session(request.session_id)
        
        # 执行对话
        response = assistant.chat(request.input)
        
        # 获取更新后的历史
        chat_history = assistant.get_chat_history()
        
        return ChatResponse(
            output=response,
            chat_history=chat_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/simple")
async def chat_simple_endpoint(request: ChatRequest):
    """
    简单对话端点 - 无工具调用，纯LLM对话
    """
    try:
        chain = create_simple_chain()
        
        # 转换历史消息格式
        history = []
        for msg in request.chat_history:
            if msg.get("role") == "user":
                history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                history.append(AIMessage(content=msg.get("content", "")))
        
        result = chain.invoke({
            "input": request.input,
            "chat_history": history
        })
        
        # 更新历史
        new_history = request.chat_history.copy()
        new_history.append({"role": "user", "content": request.input})
        new_history.append({"role": "assistant", "content": result})
        
        return ChatResponse(
            output=result,
            chat_history=new_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_endpoint(request: StreamRequest):
    """
    流式对话端点 - 支持SSE流式输出
    """
    async def generate():
        try:
            assistant = get_or_create_session(request.session_id)
            
            # 这里简化处理，实际应该使用LLM的astream方法
            response = assistant.chat(request.input)
            
            # 模拟流式输出
            words = response.split()
            for word in words:
                yield f"data: {word} "
            yield "data: [DONE]"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/session/clear")
async def clear_session_endpoint(session_id: str = "default"):
    """清除会话历史"""
    if session_id in sessions:
        sessions[session_id].clear_history()
        return {"message": f"会话 {session_id} 已清除"}
    return {"message": "会话不存在"}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史"""
    if session_id in sessions:
        return {"history": sessions[session_id].get_chat_history()}
    return {"history": []}


# ============ LangServe 路由 ============

# 添加LangChain链作为服务
simple_chain = create_simple_chain()
add_routes(
    app,
    simple_chain,
    path="/langserve",
    enabled_endpoints=["invoke", "batch", "stream"]
)


# ============ 主入口 ============

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    
    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        reload=True,
        workers=1
    )
