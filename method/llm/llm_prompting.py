from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from method.llm.models import Models
from pydantic import Field, BaseModel
from langchain.llms.base import LLM
import requests
import json
from typing import List, Dict, Optional, Union
from langchain.schema import LLMResult, Generation
import tiktoken

BASE_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
BASE_API_KEY = "8ef4c3ccf5f14ee6ad39dccaf1daef545aa3af0833ce4301a561ace8331947b2"

class CustomChatModel(LLM):
    """
    自定义模型类，用于集成私有部署的大语言模型，支持多模态输入和批量生成。
    """
    api_url: str = Field(..., description="私有模型的 API URL")
    api_key: str = Field(..., description="用于认证的 API Key")
    model: str = Field(default="gpt-4", description="使用的模型名称")
    temperature: float = Field(default=0.0, description="生成创意程度")

    def _call(self, prompt: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        """
        单条输入调用私有模型，支持消息列表输入。
        :param prompt: 用户输入的消息列表（list of BaseMessages）
        :param stop: 可选的停止标记
        :return: 模型生成的文本
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建消息格式
        messages = [{"role": message.type, "content": message.content} for message in prompt]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048,
            "stop": stop  # 如果有 stop 参数，添加到 payload 中
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"请求失败: {response.status_code}, {response.text}")

        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def _map_role(self, role: str) -> str:
        """
        将本地定义的角色映射到服务器支持的角色。
        """
        role_mapping = {
            "human": "user",       # 将 human 映射为 user
            "system": "system",    # 保持 system 不变
            "ai": "assistant",     # 将 ai 映射为 assistant
            # 如果有其他自定义角色，可在此添加映射
        }
        return role_mapping.get(role, role)  # 默认返回原值

    def generate(
        self,
        prompts: List[Union[str, Dict[str, Optional[str]]]],
        system_prompt: str = "You are a helpful assistant.",
        stop: Optional[List[str]] = None,
        callbacks: Optional[List] = None,
        **kwargs
    ) -> LLMResult:
        """
        支持字符串或字典输入的生成方法，并返回 LLMResult 对象。
        :param prompts: 字符串列表或字典列表，每个输入可以包括 system_prompt 和 user_input。
        :param stop: 可选的停止标记
        :param system_prompt: 动态设置的系统消息
        :param callbacks: 可选的回调函数列表
        :param kwargs: 捕获其他额外参数
        :return: LLMResult 对象
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        all_generations = []
        for prompt in prompts:
            if isinstance(prompt, str):
                # 如果是字符串输入
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
            elif isinstance(prompt, dict):
                # 如果是字典输入
                messages = [
                    SystemMessage(content=prompt.get("system_prompt", system_prompt)),
                    HumanMessage(content=prompt["user_input"])
                ]
                if "image_url" in prompt:
                    messages.append(HumanMessage(content=f"Image URL: {prompt['image_url']}"))
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}. Expected str or dict.")

            # 转换消息为 API 的 JSON 格式
            messages_payload = [{"role": self._map_role(message.type), "content": message.content} for message in messages]
            payload = {
                "model": self.model,
                "messages": messages_payload,
                "temperature": self.temperature,
                "max_tokens": 2048,
                "stop": stop
            }
            response = requests.post(self.api_url, headers=headers, json=payload)
            print(response.json())
            if response.status_code != 200:
                raise ValueError(f"请求失败: {response.status_code}, {response.text}")

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 将生成的内容封装为 Generation 对象
            all_generations.append([Generation(text=content)])

        # 返回 LLMResult 对象
        return LLMResult(generations=all_generations)
    @property
    def _identifying_params(self) -> Dict[str, str]:
        """
        返回模型的标识参数。
        """
        return {
            "api_url": self.api_url,
            "model": self.model,
            "temperature": str(self.temperature)
        }

    @property
    def _llm_type(self) -> str:
        """
        返回自定义 LLM 的类型标识，用于 LangChain 的内部标记。
        """
        return "custom_chat_model"
# load_dotenv()

# class CustomChatOpenAI(ChatOpenAI):
#     """
#     扩展 ChatOpenAI 类，增加 base_url 支持，用于连接自定义 OpenAI GPT 服务。
#     """
#     base_url: str = Field(default=None, description="自定义的模型服务 URL 地址")
#     def __init__(self, **kwargs):
#         """
#         初始化方法，支持自定义 base_url 和其他参数。
#         """
#         super().__init__(**kwargs)

#     def _default_openai_endpoint(self):
#         """
#         重写默认的 OpenAI 端点方法，使用自定义 base_url。
#         如果未提供 base_url，则回退到默认的 OpenAI URL。
#         """
#         return self.base_url or super()._default_openai_endpoint()


# 使用自定义类初始化


def init_model(model='gpt'):
    # if model == 'gpt_mini':
    model = CustomChatModel(api_key=BASE_API_KEY, api_url=BASE_API_URL, model="gpt-4")
    return model


def create_model_chain(model):
    def invoke_model_chain(system_prompt, user_prompt, verbose=True):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            user_prompt,
        ])
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        if model.__class__.__name__ == "custom_chat_model":
            with get_openai_callback() as cb:
                res = chain.invoke({})
                if verbose:
                    print("Response:")
                    print(res)
                    print(cb)
                    print("")
                return res
        
        return chain.invoke({})

    return invoke_model_chain


def create_single_user_message(user_prompt):
	return HumanMessage(content=[
		{
			"type": "text",
			"text": user_prompt
		}
	])


def create_multimodal_user_message(text_inputs, base64_image):
    return HumanMessage(content=[
        {
            "type": "text",
            "text": text_inputs
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            },
        }
    ])