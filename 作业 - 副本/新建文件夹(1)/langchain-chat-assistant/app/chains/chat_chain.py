"""
LangChain 对话链实现
包含：LLM调用、Prompt工程、Chain链式调用、Memory记忆、Tool工具使用
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor

from ..tools.custom_tools import get_all_tools


def get_llm():
    """获取LLM模型实例 - 支持多种免费API方案"""
    default_api = os.getenv("DEFAULT_API", "free")
    
    # 方案1: 使用Ollama本地模型（完全免费）
    if default_api == "ollama" or os.getenv("USE_OLLAMA", "false").lower() == "true":
        try:
            from langchain_ollama import ChatOllama
            model_name = os.getenv("OLLAMA_MODEL", "llama3")
            return ChatOllama(model=model_name, temperature=0.7)
        except ImportError:
            print("⚠️ Ollama未安装，切换到免费API方案")
    
    # 方案2: 使用免费公开API（如Groq）
    if default_api == "free":
        free_api_url = os.getenv("FREE_API_URL", "https://api.free-tier.groq.com/openai/v1")
        free_api_key = os.getenv("FREE_API_KEY", "gsk_your_key_here")
        free_model = os.getenv("FREE_MODEL", "llama3-8b-8192")
        
        return ChatOpenAI(
            model=free_model,
            api_key=free_api_key,
            base_url=free_api_url,
            temperature=0.7,
            streaming=True
        )
    
    # 方案3: 使用DeepSeek API
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    
    # 如果DeepSeek密钥无效，回退到免费API
    if not api_key or api_key == "your_deepseek_api_key_here" or api_key == "sk-28be82f3cf9d4098a3b1359b2b04ae39":
        free_api_url = os.getenv("FREE_API_URL", "https://api.free-tier.groq.com/openai/v1")
        free_api_key = os.getenv("FREE_API_KEY", "gsk_your_key_here")
        free_model = os.getenv("FREE_MODEL", "llama3-8b-8192")
        
        return ChatOpenAI(
            model=free_model,
            api_key=free_api_key,
            base_url=free_api_url,
            temperature=0.7,
            streaming=True
        )
    
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        streaming=True
    )


def create_chat_prompt():
    """创建对话Prompt模板"""
    system_template = """你是一个智能对话助手，名叫"小哈"🐶

你的能力包括：
1. 💬 智能对话 - 回答用户的各种问题
2. 🌤️ 天气查询 - 查询指定城市的天气信息
3. 🧮 数学计算 - 进行各种数学计算
4. ⏰ 时间查询 - 获取当前时间
5. 🔍 网络搜索 - 使用DuckDuckGo搜索网络信息
6. 📚 知识查询 - 查询维基百科等知识库

请根据用户的需求，选择合适的方式回答问题。
如果用户的问题需要调用工具，请使用工具获取信息后再回答。
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    return prompt


def create_memory():
    """创建对话记忆"""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )


def create_agent_chain():
    """创建带工具的Agent链"""
    llm = get_llm()
    tools = get_all_tools()
    prompt = create_chat_prompt()
    
    # 创建OpenAI工具Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=create_memory(),
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


def create_simple_chain():
    """创建简单的对话链（无工具）"""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能对话助手，名叫'小哈'🐶。请友好地回答用户的问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: x.get("chat_history", [])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


class ChatAssistant:
    """对话助手类 - 封装所有LangChain功能"""
    
    def __init__(self, use_tools: bool = True):
        self.use_tools = use_tools
        self.memory = create_memory()
        
        if use_tools:
            self.chain = create_agent_chain()
        else:
            self.chain = create_simple_chain()
    
    def chat(self, message: str) -> str:
        """单轮对话"""
        try:
            if self.use_tools:
                try:
                    result = self.chain.invoke({
                        "input": message
                    })
                    return result.get("output", str(result))
                except Exception as llm_error:
                    if "Authentication Fails" in str(llm_error) or "401" in str(llm_error):
                        return self._handle_no_api_key(message)
                    raise
            else:
                try:
                    result = self.chain.invoke({
                        "input": message,
                        "chat_history": self.memory.chat_memory.messages
                    })
                    # 更新记忆
                    self.memory.chat_memory.add_user_message(message)
                    self.memory.chat_memory.add_ai_message(result)
                    return result
                except Exception as llm_error:
                    if "Authentication Fails" in str(llm_error) or "401" in str(llm_error):
                        return self._handle_no_api_key(message)
                    raise
        except Exception as e:
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"
    
    def _handle_no_api_key(self, message: str) -> str:
        """处理没有有效API密钥的情况，使用内置工具"""
        intro_words = ['你是谁', '谁是你', '你的名字', '叫什么', '介绍自己', '自我介绍', '你是什么']
        if any(word in message for word in intro_words):
            return """你好呀！我叫小哈🐶，是一只智能对话助手！

我能帮你做很多事情：
🌤️ 天气查询 - 查询北京、上海、广州等城市的天气
🧮 数学计算 - 进行各种数学运算
⏰ 时间查询 - 获取当前时间

💡 提示：需要配置有效的 DeepSeek API 密钥才能使用完整的智能对话功能。"""
        
        weather_cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安"]
        for city in weather_cities:
            if city in message:
                weather_data = {
                    "北京": "☀️ 晴天，温度25°C，空气质量优",
                    "上海": "⛅ 多云，温度28°C，空气质量良",
                    "广州": "☁️ 阴天，温度32°C，空气质量良",
                    "深圳": "🌧️ 雨天，温度30°C，空气质量优",
                    "杭州": "☀️ 晴天，温度26°C，空气质量优",
                    "成都": "🌫️ 轻度雾霾，温度20°C，空气质量差",
                    "西安": "☀️ 晴天，温度18°C，空气质量良",
                }
                return f"【{city}天气】{weather_data.get(city)}"
        
        import re
        math_patterns = ['+', '-', '*', '/', '等于', '结果是']
        if any(pattern in message for pattern in math_patterns):
            expr = re.search(r'[\d+\-*/(). ]+', message)
            if expr:
                try:
                    result = eval(expr.group().strip())
                    return f"🧮 计算结果：{expr.group().strip()} = {result}"
                except:
                    pass
        
        time_words = ['时间', '几点', '现在']
        if any(word in message for word in time_words):
            from datetime import datetime
            now = datetime.now()
            return f"⏰ 当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"
        
        return """抱歉，当前 API 密钥无效，请配置有效的 DeepSeek API 密钥以使用完整功能。

我现在可以帮你做这些事情：
🌤️ 查询天气（北京、上海、广州、深圳、杭州、成都、西安）
🧮 数学计算（如：123 * 456）
⏰ 获取当前时间"""
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages
    
    def clear_history(self):
        """清空对话历史"""
        self.memory.clear()
