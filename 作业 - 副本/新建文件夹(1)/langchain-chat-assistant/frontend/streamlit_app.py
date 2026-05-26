"""
Streamlit 前端界面 - LangChain对话助手
支持连接LangServe后端
"""

import os
import sys
import uuid
import requests
from typing import List, Dict
from datetime import datetime

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="小哈 - LangChain智能助手",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    /* 状态指示器 */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
    }
    
    .status-offline {
        background-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ============ 会话状态初始化 ============

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")

if "api_configured" not in st.session_state:
    st.session_state.api_configured = False

# ============ 工具函数 ============

def check_api_connection() -> bool:
    """检查API连接状态"""
    try:
        response = requests.get(
            f"{st.session_state.api_url}/health",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.api_configured = data.get("api_configured", False)
            return True
        return False
    except Exception:
        return False


def send_message(message: str) -> Dict:
    """发送消息到后端"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/chat",
            json={
                "input": message,
                "chat_history": st.session_state.messages,
                "session_id": st.session_state.session_id
            },
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"output": f"连接错误：{str(e)}。请检查后端服务是否运行。", "chat_history": []}


def clear_chat():
    """清空对话"""
    st.session_state.messages = []
    try:
        requests.post(
            f"{st.session_state.api_url}/session/clear",
            params={"session_id": st.session_state.session_id}
        )
    except:
        pass

# ============ 侧边栏 ============

with st.sidebar:
    # Logo和标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🐶</div>
        <h1 style="color: white; margin: 0;">小哈助手</h1>
        <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">LangChain 智能对话</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API配置
    st.markdown("---")
    st.subheader("⚙️ 后端配置")
    
    api_url = st.text_input(
        "API地址",
        value=st.session_state.api_url,
        placeholder="http://localhost:8000"
    )
    st.session_state.api_url = api_url
    
    # 连接状态
    if st.button("🔄 检查连接"):
        with st.spinner("检查中..."):
            if check_api_connection():
                st.success("✅ 后端连接正常")
            else:
                st.error("❌ 无法连接后端")
    
    # 功能介绍
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
    
    # 操作按钮
    st.markdown("---")
    if st.button("🗑️ 清空对话", use_container_width=True):
        clear_chat()
        st.rerun()
    
    # 关于
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
        <p>基于 LangChain 构建</p>
        <p>集成 LangSmith 监控</p>
        <p>LangServe 部署</p>
    </div>
    """, unsafe_allow_html=True)

# ============ 主界面 ============

# 标题区域
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🐶 小哈智能助手</h1>
    <p style="color: #6b7280;">基于 LangChain 的智能对话助手，支持工具调用和记忆功能</p>
</div>
""", unsafe_allow_html=True)

# 连接状态提示
if not check_api_connection():
    st.warning("⚠️ 后端服务未连接，请在侧边栏配置正确的API地址")

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
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 显示思考中
        with st.spinner("小哈思考中..."):
            response = send_message(user_input)
        
        # 添加助手回复
        output = response.get("output", "抱歉，处理出错")
        st.session_state.messages.append({"role": "assistant", "content": output})
        
        # 清空当前输入
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
