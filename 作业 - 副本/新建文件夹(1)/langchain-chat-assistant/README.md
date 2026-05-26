# 🐶 小哈智能助手 - LangChain Chat Assistant

基于 LangChain 框架开发的智能对话助手，集成 LangSmith 监控、LangServe 部署和多种实用工具。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)

## ✨ 核心功能

### 🤖 LangChain 核心要素

- **LLM调用**：支持 DeepSeek / OpenAI 等大语言模型
- **Prompt工程**：结构化提示词模板设计
- **Chain链式调用**：多步骤处理流程编排
- **Memory记忆**：对话历史上下文保持
- **Tool工具使用**：天气、计算、搜索等实用工具

### 🛠️ 支持工具

| 工具 | 功能 | 示例 |
|------|------|------|
| 🌤️ 天气查询 | 查询城市天气 | "北京天气怎么样？" |
| 🧮 数学计算 | 执行数学运算 | "计算 123 * 456" |
| ⏰ 时间查询 | 获取当前时间 | "现在几点了？" |
| 🔍 网络搜索 | DuckDuckGo搜索 | "搜索 LangChain 最新版本" |
| 💬 智能对话 | 自然语言交流 | "你是谁？" |

### 📊 监控与部署

- **LangSmith**：全流程追踪和监控
- **LangServe**：高并发API服务部署
- **Streamlit Cloud**：免费云端部署

## 🏗️ 架构设计

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   LangServe     │────▶│   LangChain     │
│   Frontend      │◀────│   Backend       │◀────│   Core          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LangSmith     │     │   LLM / Tools   │
                        │   Monitoring    │     │   Memory        │
                        └─────────────────┘     └─────────────────┘
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/langchain-chat-assistant.git
cd langchain-chat-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 4. 启动服务

**启动后端（LangServe）：**
```bash
python -m app.server
# 或
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

**启动前端（Streamlit）：**
```bash
streamlit run frontend/streamlit_app.py
```

访问地址：
- 后端API文档：http://localhost:8000/docs
- 前端界面：http://localhost:8501

## 📁 项目结构

```
langchain-chat-assistant/
├── app/
│   ├── __init__.py
│   ├── server.py              # LangServe 后端服务
│   ├── chains/
│   │   ├── __init__.py
│   │   └── chat_chain.py      # LangChain 对话链实现
│   └── tools/
│       ├── __init__.py
│       └── custom_tools.py    # 自定义工具集
├── frontend/
│   ├── __init__.py
│   └── streamlit_app.py       # Streamlit 前端
├── .env.example               # 环境变量示例
├── requirements.txt           # 依赖列表
├── README.md                  # 项目说明
└── packages.txt               # Streamlit Cloud 系统依赖
```

## ⚙️ 配置说明

### 必需配置

```env
# DeepSeek API（推荐）
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 或 OpenAI API
OPENAI_API_KEY=your_openai_api_key
```

### 可选配置

```env
# LangSmith 监控（推荐）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=langchain-chat-assistant

# 应用配置
APP_HOST=0.0.0.0
APP_PORT=8000
FRONTEND_PORT=8501
MODEL_NAME=deepseek-chat
```

## 🌐 部署指南

### Streamlit Cloud 部署（推荐）

1. **Fork 本项目** 到你的GitHub账号

2. **登录 [Streamlit Cloud](https://streamlit.io/cloud)**

3. **创建新应用**
   - Repository: 选择你的fork
   - Branch: `main`
   - Main file path: `frontend/streamlit_app.py`

4. **配置 Secrets**
   在 Streamlit Cloud 的 Settings → Secrets 中添加：
   ```toml
   DEEPSEEK_API_KEY = "your_api_key"
   LANGCHAIN_API_KEY = "your_langsmith_key"
   ```

5. **部署完成！**

### 其他部署方式

#### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Railway / Render 部署

1. 连接 GitHub 仓库
2. 设置环境变量
3. 自动部署

## 📚 API 文档

启动后端后访问：`http://localhost:8000/docs`

### 主要接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 对话接口（支持工具） |
| `/chat/simple` | POST | 简单对话接口 |
| `/chat/stream` | POST | 流式对话接口 |
| `/health` | GET | 健康检查 |
| `/session/clear` | POST | 清除会话 |

### 请求示例

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "北京天气怎么样？",
    "session_id": "user-123"
  }'
```

## 📊 LangSmith 监控

配置 `LANGCHAIN_API_KEY` 后，所有调用将自动追踪：

- ✅ LLM 调用记录
- ✅ 工具调用追踪
- ✅ 链式执行流程
- ✅ Token 使用量统计
- ✅ 延迟和性能指标

访问 [LangSmith](https://smith.langchain.com) 查看监控数据。

## 🤝 贡献指南

1. Fork 本项目
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://langchain.com/) - 大模型应用框架
- [LangServe](https://github.com/langchain-ai/langserve) - 部署工具
- [LangSmith](https://smith.langchain.com/) - 监控平台
- [Streamlit](https://streamlit.io/) - 前端框架

---

⭐ 如果这个项目对你有帮助，请给它一个 Star！
