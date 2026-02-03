# one_eval/serving/custom_llm_caller.py
from __future__ import annotations
import httpx
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI

from dataflow_agent.llm_callers.base import BaseLLMCaller
from one_eval.logger import get_logger

log = get_logger("CustomLLMCaller")


class CustomLLMCaller(BaseLLMCaller):
    """
    ✔ 完全兼容 BaseAgent / Tool / LangGraph 的 LLM 调用器
    ✔ 支持 bind_tools(自动工具调用)
    ✔ 直接用 httpx 调用你的 API(不依赖 ChatOpenAI 的网络请求)
    """

    def __init__(
        self,
        state,
        tool_manager,
        agent_role: str,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0
    ):
        super().__init__(
            state=state,
            tool_manager=tool_manager,
            model_name=model_name,
            temperature=temperature,
            )

        self.agent_role = agent_role   # 保存 agent 的真实角色名
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=60,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    # ------------------------------
    #  基础 API 调用（最快）
    # ------------------------------

    def _convert_lc_message(self, m: BaseMessage):
        """
        将 LangChain 的 Message 转换为 OpenAI API 支持的消息格式。
        """

        # 1) ToolMessage
        if m.type == "tool":
            return {
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": str(m.content),
            }

        # 2) AIMessage with tool_calls (模型在第一轮要求调用工具)
        if isinstance(m, AIMessage) and m.additional_kwargs.get("tool_calls"):
            return {
                "role": "assistant",
                "tool_calls": m.additional_kwargs["tool_calls"],
                # 不能填 content
            }

        # 3) Normal assistant message (must map type → "assistant")
        if isinstance(m, AIMessage):
            return {
                "role": "assistant",
                "content": m.content or "",
            }

        # 4) HumanMessage
        if m.type == "human":
            return {
                "role": "user",
                "content": m.content,
            }

        # 5) SystemMessage
        if m.type == "system":
            return {
                "role": "system",
                "content": m.content,
            }

        # 6) fallback
        return {
            "role": "assistant",
            "content": m.content or "",
        }

    async def _call_raw_api(self, messages: List[BaseMessage]) -> AIMessage:
        api_url = f"{self.base_url}/chat/completions"

        formatted_messages = [self._convert_lc_message(m) for m in messages]

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
        }

        r = await self._client.post(api_url, json=payload)
        r.raise_for_status()
        data = r.json()

        content = data["choices"][0]["message"].get("content", "")
        return AIMessage(content=content)


    # ------------------------------
    #  LLM 调用入口（框架使用）
    # ------------------------------
    async def call(self, messages: List[BaseMessage], bind_post_tools: bool) -> AIMessage:
        """
        bind_post_tools = False → 直接调 API，性能最高
        bind_post_tools = True → 用 ChatOpenAI + bind_tools（兼容 LangGraph 工具链）
        """

        # =====================================================
        # 1) 无工具模式 —— 使用你自己的 API（这是大多数情况）
        # =====================================================
        if not bind_post_tools:
            return await self._call_raw_api(messages)

        # =====================================================
        # 2) 工具模式 —— 用 ChatOpenAI.bind_tools
        # =====================================================
        post_tools = []
        if self.tool_manager:
            post_tools = self.tool_manager.get_post_tools(self.agent_role)

        log.info(f"[CustomLLMCaller] Binding {len(post_tools)} tools")

        llm = ChatOpenAI(
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        ).bind_tools(post_tools)

        return await llm.ainvoke(messages)

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """兼容 LangChain Runnable 接口"""
        # 默认这里只做简单的对话，不绑定工具
        # 如果需要工具，Agent 层应该显式调用 call(bind_post_tools=True)
        return await self.call(messages, bind_post_tools=False)


class EmbeddingCaller:
    """
    Embedding 调用器，复用 LLM 的 API 配置。
    用于 RAG 场景下的文本向量化。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的 embedding 向量

        Args:
            texts: 文本列表

        Returns:
            embedding 向量列表
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        # 按 index 排序确保顺序正确
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def get_embedding_batch(
        self,
        texts: List[str],
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        批量获取 embedding（适用于大量文本）

        Args:
            texts: 文本列表
            batch_size: 每批大小

        Returns:
            embedding 向量列表
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embedding(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
