from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from typing import Any, Dict, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import os

from one_eval.serving.custom_llm_caller import CustomLLMCaller
from one_eval.utils.prompts import prompt_registry
from one_eval.logger import get_logger

log = get_logger(__name__)

# 验证器类型定义：返回 (是否通过, 错误信息)
ValidatorFunc = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]]


class BaseAgent(ABC):
    """Agent基类 - 定义通用的agent执行模式"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            tmp = cls(tool_manager=None)          # BaseAgent 的 __init__ 很轻
            name = tmp.role_name
            # 移除AgentRegistry注册，避免循环引用
        except Exception as e:
            pass
    
    def __init__(self, 
                 tool_manager: Optional[Any] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 tool_mode: str = "auto",
                 react_mode: bool = False,
                 react_max_retries: int = 3,
                 # 新增参数
                 parser_type: str = "json",
                 parser_config: Optional[Dict[str, Any]] = None,
                 use_vlm: bool = False,
                 vlm_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            parser_type: 解析器类型 ("json", "xml", "text")
            parser_config: 解析器配置（如XML的root_tag）
            use_vlm: 是否使用视觉语言模型
            vlm_config: VLM配置字典
        """
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.react_mode = react_mode
        self.react_max_retries = react_max_retries
        
        # 解析器配置
        self.parser_type = parser_type
        self.parser_config = parser_config or {}
        self._parser = None  # 懒加载
        
        # VLM配置
        self.use_vlm = use_vlm
        self.vlm_config = vlm_config or {}

    @classmethod
    def create(cls, tool_manager: Optional[Any] = None, **kwargs) -> "BaseAgent":
        """
        工厂方法：保持所有 Agent 统一的创建入口。
        """
        return cls(tool_manager=tool_manager, **kwargs)
    
    @property
    @abstractmethod
    def role_name(self) -> str:
        """角色名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def system_prompt_template_name(self) -> str:
        """系统提示词模板名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def task_prompt_template_name(self) -> str:
        """任务提示词模板名称 - 子类必须实现"""
        pass
    
    def parse_result(self, content: str) -> Dict[str, Any]:
        """解析结果 - 基础JSON解析实现"""
        try:
            import json
            # 尝试提取JSON内容
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            parsed = json.loads(content)
            # log.info(f"{self.role_name} 结果解析成功")
            return parsed
        except ValueError as e:
            log.warning(f"JSON解析失败: {e}")
            return {"raw": content}
        except Exception as e:
            log.warning(f"解析过程出错: {e}")
            return {"raw": content}
    
    def update_state_result(self, state: Any, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """更新状态结果"""
        state.result = result


class CustomAgent(BaseAgent):
    """
    CustomAgent
    ----------
    继承自 BaseAgent
    在 one_eval 框架中用于执行单个节点逻辑。

    处理：
        - 执行前置工具
        - 调用 LLM
        - 更新状态结果
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = os.getenv(
            "OE_API_BASE",
            "http://123.129.219.111:3000/v1",
        )
        self.api_key = os.getenv(
            "OE_API_KEY",
            "sk-xxxxx",
        )

    # ======= 必须实现的抽象属性 =======
    @property
    def role_name(self) -> str:
        return "CustomAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_default"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_default"

    def create_llm(self, state):
        return CustomLLMCaller(
            state=state,
            tool_manager=self.tool_manager,
            model_name=self.model_name,
            base_url=self.api_url,
            api_key=self.api_key,
            agent_role=self.role_name,
        )

    def get_prompt(self, name: str, **kwargs) -> str:
        """统一获取 prompt"""
        tmpl = prompt_registry.get(name)
        return tmpl.build_prompt(**kwargs)

    # ======= 核心执行逻辑 =======
    async def run(self, state):
        pre = {}
        msgs = [
            SystemMessage(content=self.get_prompt(self.system_prompt_template_name)),
            HumanMessage(content=self.get_prompt(self.task_prompt_template_name, **state.__dict__))
        ]
        llm = self.create_llm(state)

        resp = await llm.ainvoke(msgs)
        result = self.parse_result(resp.content)

        # 写入 state
        self.update_state_result(state, result, pre)
        return state

    # ======= 可选：自定义解析逻辑 =======
    # def parse_result(self, content: str) -> dict[str, Any]:
    #     """可重写解析逻辑(默认复用 BaseAgent 的 robust_parse_json)"""
    #     result = super().parse_result(content)
    #     if "raw" in result:
    #         log.warning("[CustomAgent] LLM 输出未解析为 JSON, 使用 raw 内容。")
    #     return result
