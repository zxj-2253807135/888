"""
自定义工具集 - LangChain Tool工具使用示例
"""

import os
import re
from datetime import datetime
from typing import Type, Optional

from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS


# ============ 工具输入模型定义 ============

class WeatherInput(BaseModel):
    """天气查询工具输入"""
    city: str = Field(description="要查询天气的城市名称，如'北京'、'上海'等")


class CalculatorInput(BaseModel):
    """计算器工具输入"""
    expression: str = Field(description="数学表达式，如'2 + 3 * 4'、'sqrt(16)'等")


class SearchInput(BaseModel):
    """网络搜索工具输入"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=3, description="返回结果数量")


# ============ 工具实现 ============

class WeatherTool(BaseTool):
    """天气查询工具"""
    name: str = "weather"
    description: str = "查询指定城市的天气信息。输入城市名称，返回天气情况。"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        """执行天气查询"""
        weather_data = {
            "北京": "☀️ 晴天，温度25°C，空气质量优，适宜户外活动",
            "上海": "⛅ 多云，温度28°C，空气质量良，建议携带雨具",
            "广州": "☁️ 阴天，温度32°C，空气质量良，注意防暑",
            "深圳": "🌧️ 小雨，温度30°C，空气质量优，出门请带伞",
            "杭州": "☀️ 晴天，温度26°C，空气质量优，西湖风景宜人",
            "成都": "🌫️ 轻度雾霾，温度20°C，空气质量差，建议减少外出",
            "西安": "☀️ 晴天，温度18°C，空气质量良，适合游览古迹",
            "武汉": "⛈️ 雷阵雨，温度27°C，空气质量良，注意防雷",
            "南京": "🌤️ 晴转多云，温度24°C，空气质量优",
            "重庆": "☁️ 阴天，温度29°C，空气质量良，火锅天气",
        }
        
        result = weather_data.get(city)
        if result:
            return f"【{city}天气】{result}"
        return f"抱歉，暂未获取到{city}的天气信息。支持查询的城市：{', '.join(weather_data.keys())}"


class CalculatorTool(BaseTool):
    """数学计算工具"""
    name: str = "calculator"
    description: str = "执行数学计算。输入数学表达式，返回计算结果。支持加减乘除、括号等运算。"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """执行计算"""
        try:
            # 清理表达式，只允许安全字符
            cleaned = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            if not cleaned:
                return "错误：无效的数学表达式"
            
            # 安全计算
            result = eval(cleaned)
            return f"🧮 计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算错误：{str(e)}。请检查表达式格式。"


class TimeTool(BaseTool):
    """时间查询工具"""
    name: str = "current_time"
    description: str = "获取当前日期和时间。无需输入参数。"
    
    def _run(self) -> str:
        """获取当前时间"""
        now = datetime.now()
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
        return f"⏰ 当前时间：{now.strftime('%Y年%m月%d日')} {weekday} {now.strftime('%H:%M:%S')}"


class DuckDuckGoSearchTool(BaseTool):
    """DuckDuckGo网络搜索工具"""
    name: str = "web_search"
    description: str = "使用DuckDuckGo搜索引擎查询网络信息。输入搜索关键词，返回搜索结果。"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 3) -> str:
        """执行网络搜索"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
                if not results:
                    return f"未找到关于'{query}'的搜索结果。"
                
                output = [f"🔍 搜索结果：'{query}'\n"]
                for i, result in enumerate(results, 1):
                    title = result.get('title', '无标题')
                    href = result.get('href', '')
                    body = result.get('body', '无摘要')
                    output.append(f"{i}. {title}\n   {body[:100]}...\n   链接：{href[:60]}...\n")
                
                return "\n".join(output)
        except Exception as e:
            return f"搜索出错：{str(e)}。请稍后重试。"


class IntroductionTool(BaseTool):
    """自我介绍工具"""
    name: str = "self_introduction"
    description: str = "当用户询问'你是谁'、'自我介绍'等问题时，返回助手的自我介绍。"
    
    def _run(self) -> str:
        """返回自我介绍"""
        return """你好呀！我叫**小哈**🐶，是一只智能对话助手！

我能帮你做很多事情：

🌤️ **天气查询** - 查询北京、上海、广州等城市的天气
🧮 **数学计算** - 加减乘除、复杂运算都不在话下  
⏰ **时间查询** - 随时告诉你现在几点了
🔍 **网络搜索** - 帮你搜索最新的网络信息
💬 **智能对话** - 陪你聊天，解答各种问题

我基于 **LangChain** 框架开发，集成了：
- 🤖 大语言模型（LLM）调用
- 📝 Prompt工程
- 🔗 Chain链式调用
- 🧠 Memory记忆功能
- 🛠️ Tool工具使用

有什么可以帮助你的吗？随时告诉我！"""


# ============ 工具集合 ============

def get_all_tools() -> list:
    """获取所有可用工具"""
    return [
        WeatherTool(),
        CalculatorTool(),
        TimeTool(),
        DuckDuckGoSearchTool(),
        IntroductionTool(),
    ]


def get_tool_descriptions() -> str:
    """获取工具描述信息"""
    tools = get_all_tools()
    descriptions = []
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)
