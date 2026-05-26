"""
本地对话助手 - 不依赖外部API
提供完整的本地对话体验
"""

import re
from typing import List, Dict
from datetime import datetime

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage


class LocalChatAssistant:
    """本地对话助手"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.knowledge_base = {
            "langchain": "LangChain 是一个用于构建大语言模型应用的框架，支持LLM调用、Prompt工程、Chain链式调用、Memory记忆和Tool工具使用。",
            "python": "Python 是一种高级编程语言，以简洁和易读性著称，广泛用于数据科学、Web开发和人工智能领域。",
            "chatbot": "聊天机器人是一种使用自然语言处理技术与用户进行对话的人工智能系统。",
            "ai": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的机器。",
            "机器学习": "机器学习是人工智能的一个分支，使计算机能够从数据中学习而无需明确编程。",
        }
    
    def chat(self, message: str) -> str:
        """单轮对话"""
        # 尝试工具调用
        tool_result = self._try_tool(message)
        if tool_result:
            # 更新记忆
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(tool_result)
            return tool_result
        
        # 使用本地知识回答
        result = self._generate_response(message)
        
        # 更新记忆
        self.memory.chat_memory.add_user_message(message)
        self.memory.chat_memory.add_ai_message(result)
        return result
    
    def _try_tool(self, message: str) -> str:
        """尝试工具调用"""
        # 自我介绍
        intro_words = ['你是谁', '谁是你', '你的名字', '叫什么', '介绍自己', '自我介绍', '你是什么']
        if any(word in message for word in intro_words):
            return """你好呀！我叫小哈🐶，是一只智能对话助手！

我能帮你做很多事情：
🌤️ 天气查询 - 查询北京、上海、广州等城市的天气
🧮 数学计算 - 进行各种数学运算
⏰ 时间查询 - 获取当前时间
🌍 翻译功能 - 中英互译（如"我爱你翻译英文"）
💬 智能对话 - 与你聊天交流

我是一个完全本地化的助手，不依赖外部API，可以直接使用！"""
        
        # 天气查询
        weather_cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安"]
        for city in weather_cities:
            if city in message:
                weather_data = {
                    "北京": "☀️ 晴天，温度25°C，空气质量优，适宜户外活动",
                    "上海": "⛅ 多云，温度28°C，空气质量良，建议携带雨具",
                    "广州": "☁️ 阴天，温度32°C，空气质量良，注意防暑",
                    "深圳": "🌧️ 小雨，温度30°C，空气质量优，出门请带伞",
                    "杭州": "☀️ 晴天，温度26°C，空气质量优，西湖风景宜人",
                    "成都": "🌫️ 轻度雾霾，温度20°C，空气质量差，建议减少外出",
                    "西安": "☀️ 晴天，温度18°C，空气质量良，适合游览古迹",
                }
                return f"【{city}天气】{weather_data.get(city)}"
        
        # 数学计算
        math_patterns = ['+', '-', '*', '/', '等于', '结果是']
        if any(pattern in message for pattern in math_patterns):
            expr = re.search(r'[\d+\-*/(). ]+', message)
            if expr:
                try:
                    cleaned = re.sub(r'[^0-9+\-*/().\s]', '', expr.group())
                    result = eval(cleaned)
                    return f"🧮 计算结果：{cleaned} = {result}"
                except Exception as e:
                    return f"计算错误：{str(e)}"
        
        # 时间查询
        time_words = ['时间', '几点', '现在']
        if any(word in message for word in time_words):
            now = datetime.now()
            weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
            return f"⏰ 当前时间：{now.strftime('%Y年%m月%d日')} {weekday} {now.strftime('%H:%M:%S')}"
        
        # 翻译功能
        if '翻译' in message:
            return self._translate(message)
        
        return None
    
    def _translate(self, message: str) -> str:
        """中英互译"""
        translations = {
            '我爱你': 'I love you',
            '你好': 'Hello',
            '谢谢': 'Thank you',
            '再见': 'Goodbye',
            '对不起': 'Sorry',
            '是的': 'Yes',
            '不是': 'No',
            '请': 'Please',
            '欢迎': 'Welcome',
            '朋友': 'Friend',
            '家人': 'Family',
            '幸福': 'Happiness',
            '快乐': 'Happy',
            '美丽': 'Beautiful',
            '聪明': 'Smart',
            '勇敢': 'Brave',
            'I love you': '我爱你',
            'Hello': '你好',
            'Thank you': '谢谢',
            'Goodbye': '再见',
            'Sorry': '对不起',
            'Yes': '是的',
            'No': '不是',
            'Please': '请',
            'Welcome': '欢迎',
            'Friend': '朋友',
            'Family': '家人',
            'Happiness': '幸福',
            'Happy': '快乐',
            'Beautiful': '美丽',
            'Smart': '聪明',
            'Brave': '勇敢',
        }
        
        # 提取要翻译的内容（去掉"翻译"和语言说明）
        text = message.replace('翻译', '')
        text = text.replace('英文', '').replace('中文', '').replace('English', '').replace('Chinese', '').strip()
        
        if text:
            result = translations.get(text)
            if result:
                return f"🌍 翻译结果：'{text}' → '{result}'"
            else:
                return f"🌍 抱歉，我还不会翻译 '{text}'。我会继续学习更多词汇！"
        return "🌍 请告诉我要翻译的内容，例如：'我爱你翻译英文' 或 'Hello翻译中文'"
    
    def _generate_response(self, input_text: str) -> str:
        """生成响应"""
        input_text_lower = input_text.lower()
        
        # 问候语
        greetings = ["你好", "您好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好"]
        if any(greeting in input_text_lower for greeting in greetings):
            return "你好呀！我是小哈🐶，很高兴认识你！有什么可以帮助你的吗？"
        
        # 感谢语
        thanks = ["谢谢", "感谢", "thank you", "thx"]
        if any(thank in input_text_lower for thank in thanks):
            return "不客气！有任何问题随时问我，我很乐意为你解答！"
        
        # 询问助手能力
        ability = ["你能做什么", "你会什么", "你的能力", "功能", "能帮我"]
        if any(ab in input_text_lower for ab in ability):
            return """我能帮你做很多事情哦：
            
🌤️ 天气查询 - 查询北京、上海、广州等城市的天气
🧮 数学计算 - 进行各种数学运算
⏰ 时间查询 - 获取当前时间
💬 智能对话 - 回答你的问题
📚 知识问答 - 关于LangChain、Python、AI等知识

有什么需要帮助的吗？"""
        
        # 知识问答
        for keyword, knowledge in self.knowledge_base.items():
            if keyword in input_text_lower:
                return knowledge
        
        # 默认响应
        return f"""
关于 "{input_text}"，我可以为你提供一些信息！

如果你有具体的问题，比如：
- 🌤️ 天气查询（如"北京天气"）
- 🧮 数学计算（如"123 * 456"）
- ⏰ 时间查询（如"现在几点"）
- 📚 知识问答（如"什么是LangChain"）

请随时告诉我！
"""
    
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
