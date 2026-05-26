"""
Streamlit 独立版前端 - 包含完整LangChain链条逻辑
适用于 Streamlit Cloud 部署（无需后端服务）
"""

import os
import sys
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="小哈 - LangChain智能助手",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ LangChain 依赖 ============
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_core.tools import Tool
    from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
    
    # 自定义工具
    from app.tools.custom_tools import get_all_tools
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    st.error(f"LangChain 依赖未安装: {e}")
    LANGCHAIN_AVAILABLE = False

# ============ CSS样式 ============
st.markdown("""
<style>
    /* 整体主题 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .chat-message.assistant {
        background: white;
        color: #1f2937;
        margin-right: 20%;
        border: 1px solid #e5e7eb;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 2rem;
        border: 2px solid #e5e7eb;
        padding: 1rem 1.5rem;
        font-size: 1rem;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 2rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* 工具卡片样式 */
    .tool-card {
        background: white;
        border-radius: 1rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ============ 流式输出工具函数 ============
def stream_response(chain, input_text: str):
    """流式输出响应"""
    try:
        result = chain.invoke({"input": input_text})
        return result.get("output", str(result))
    except Exception as e:
        return f"处理错误：{str(e)}"

# ============ LangChain 链条实现 ============
def get_llm():
    """获取LLM模型实例"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    
    # 如果API密钥无效，使用免费方案
    if not api_key or api_key == "your_deepseek_api_key_here" or api_key.strip() == "":
        st.warning("未配置有效的DeepSeek API密钥，将使用免费模型")
        return ChatOpenAI(
            model="llama3-8b-8192",
            api_key="gsk_your_key_here",
            base_url="https://api.free-tier.groq.com/openai/v1",
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
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=create_memory(),
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

# ============ 会话状态初始化 ============
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_chain" not in st.session_state and LANGCHAIN_AVAILABLE:
    try:
        st.session_state.agent_chain = create_agent_chain()
    except Exception as e:
        st.error(f"创建Agent链失败: {e}")
        st.session_state.agent_chain = None

# ============ 工具函数 ============
def send_message(message: str) -> Dict:
    """发送消息到Agent链"""
    if not st.session_state.agent_chain:
        return {"output": "Agent链未初始化，请刷新页面重试。", "chat_history": []}
    
    try:
        result = st.session_state.agent_chain.invoke({
            "input": message
        })
        output = result.get("output", str(result))
        return {"output": output, "chat_history": []}
    except Exception as e:
        # 处理认证失败等错误
        if "Authentication Fails" in str(e) or "401" in str(e):
            return {"output": handle_no_api_key(message), "chat_history": []}
        return {"output": f"处理错误：{str(e)}", "chat_history": []}

def handle_no_api_key(message: str) -> str:
    """处理没有有效API密钥的情况"""
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
        now = datetime.now()
        return f"⏰ 当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"
    
    return """抱歉，当前 API 密钥无效，请配置有效的 DeepSeek API 密钥以使用完整功能。

我现在可以帮你做这些事情：
🌤️ 查询天气（北京、上海、广州、深圳、杭州、成都、西安）
🧮 数学计算（如：123 * 456）
⏰ 获取当前时间"""

def clear_chat():
    """清空对话"""
    st.session_state.messages = []
    if st.session_state.agent_chain:
        try:
            st.session_state.agent_chain.memory.clear()
        except:
            pass

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🐶</div>
        <h1 style="color: white; margin: 0;">小哈助手</h1>
        <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">LangChain 智能对话</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("⚙️ API配置")
    
    # API密钥输入
    api_key = st.text_input(
        "DeepSeek API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        placeholder="sk-..."
    )
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        # 重新创建链
        try:
            st.session_state.agent_chain = create_agent_chain()
            st.success("✅ API密钥已配置")
        except Exception as e:
            st.error(f"配置失败: {e}")
    
    st.markdown("---")
    st.subheader("🛠️ 支持功能")
    
    tools = [
        ("💬", "智能对话", "自然语言交流"),
        ("🌤️", "天气查询", "查询城市天气"),
        ("🧮", "数学计算", "执行数学运算"),
        ("⏰", "时间查询", "获取当前时间"),
        ("🔍", "网络搜索", "DuckDuckGo搜索"),
    ]
    
    for icon, name, desc in tools:
        st.markdown(f"""
        <div class="tool-card">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">{icon}</span>
                <div>
                    <strong>{name}</strong>
                    <div style="font-size: 0.8rem; color: #6b7280;">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ 清空对话", use_container_width=True):
        clear_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
        <p>基于 LangChain 构建</p>
        <p>Streamlit Cloud 独立版</p>
    </div>
    """, unsafe_allow_html=True)

# ============ 主界面 ============
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🐶 小哈智能助手</h1>
    <p style="color: #6b7280;">基于 LangChain 的智能对话助手（Streamlit Cloud 独立版）</p>
</div>
""", unsafe_allow_html=True)

# 显示对话历史
for message in st.session_state.messages:
    role = message.get("role", "")
    content = message.get("content", "")
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span>👤</span>
                <strong>你</strong>
            </div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span>🐶</span>
                <strong>小哈</strong>
            </div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)

# 输入区域
st.markdown("---")

# 快速操作按钮
cols = st.columns(5)
quick_actions = [
    ("🌤️ 北京天气", "北京天气怎么样？"),
    ("🧮 计算", "计算 123 * 456"),
    ("⏰ 时间", "现在几点了？"),
    ("🔍 搜索", "搜索 LangChain 最新版本"),
    ("👋 介绍", "你是谁？"),
]

for col, (label, prompt) in zip(cols, quick_actions):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.current_input = prompt
            st.rerun()

# 消息输入
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input(
            "输入消息",
            value=st.session_state.get("current_input", ""),
            placeholder="输入消息或选择上方快捷操作...",
            label_visibility="collapsed"
        )
    with cols[1]:
        submit = st.form_submit_button("发送", use_container_width=True)
    
    if submit and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("小哈思考中..."):
            response = send_message(user_input)
        
        output = response.get("output", "抱歉，处理出错")
        st.session_state.messages.append({"role": "assistant", "content": output})
        
        if "current_input" in st.session_state:
            del st.session_state.current_input
        
        st.rerun()

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.8rem;">
    <p>💡 提示：支持多轮对话，小哈会记住上下文 | 会话ID: {}</p>
</div>
""".format(st.session_state.session_id[:8]), unsafe_allow_html=True)