from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from one_eval.toolkits.tool_manager import get_tool_manager
from typing import Optional, Callable, Dict, List, Tuple, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
import asyncio

class GraphBuilder(GenericGraphBuilder):
    """Eval流程Graph的构建类"""
    def __init__(self, state_model, entry_point="start"):
        # === 基础图结构 ===
        self.state_model = state_model
        self.entry_point = entry_point
        self.nodes: Dict[str, Tuple[Callable, str]] = {}   # 节点名 → (函数, 角色)
        self.edges: List[Tuple[str, str]] = []             # 普通边
        self.conditional_edges: Dict[str, Callable] = {}   # 条件边
        
        # === 工具注册（简化版，仅 Tool）===
        self.tool_registry: Dict[str, Callable] = {}       # 工具名 → 函数
        
        # === 兼容字段（保留以防后续拓展）===
        self.pre_tool_registry: Dict[str, Dict[str, Callable]] = {}
        self.post_tool_registry: Dict[str, List[Callable]] = {}
        self.tool_manager = None                           # 延迟导入

    # 用类型注释显式暴露父类方法，便于自动补全和文档生成
    def add_node(self, name: str, func: Callable, role: str | None = None) -> "GraphBuilder":
        super().add_node(name, func, role)
        return self
        
    def add_nodes(self, nodes: Dict[str, Callable], role_mapping: Dict[str, str] | None = None) -> "GraphBuilder":
        super().add_nodes(nodes, role_mapping)
        return self
        
    def add_edge(self, src: str, dst: str) -> "GraphBuilder":
        super().add_edge(src, dst)
        return self
        
    def add_edges(self, edges: List[Tuple[str, str]]) -> "GraphBuilder":
        super().add_edges(edges)
        return self
        
    def add_conditional_edge(self, src: str, condition_func: Callable) -> "GraphBuilder":
        super().add_conditional_edge(src, condition_func)
        return self
        
    def add_conditional_edges(self, conditional_edges: Dict[str, Callable]) -> "GraphBuilder":
        super().add_conditional_edges(conditional_edges)
        return self

    def custom_tool(self, name: str, role: str):
        def decorator(func):
            if not hasattr(self, "custom_tool_registry"):
                self.custom_tool_registry = {}
            if role not in self.custom_tool_registry:
                self.custom_tool_registry[role] = {}
            self.custom_tool_registry[role][name] = func
            return func
        return decorator

    def _register_tools_for_role(self, role: str, state: Any):
        """
        为指定角色注册自定义工具(不区分 pre/post)。
        支持 override 兼容。
        """
        tm = self._get_tool_manager()
        if not hasattr(self, "custom_tool_registry"):
            return

        if role in self.custom_tool_registry:
            for tool_name, tool_func in self.custom_tool_registry[role].items():
                try:
                    tm.register_custom_tool(
                        name=tool_name,
                        role=role,
                        func=lambda s=state, f=tool_func: f(s),
                        override=True,
                    )
                except TypeError:
                    tm.register_custom_tool(
                        name=tool_name,
                        role=role,
                        func=lambda s=state, f=tool_func: f(s),
                    )

    def _wrap_node_with_tools(self, node_func: Callable, role: str):
        """
        重写父类的包装器。
        原因：当启用 checkpointer 时，LangGraph 可能会向节点传递 `config` 等额外参数。
        父类的包装器只接受 `state`，会导致 TypeError。
        """
        async def wrapped_node(state, config: RunnableConfig = None):
            token = None
            if config is not None:
                try:
                    token = var_child_runnable_config.set(config)
                except Exception:
                    token = None
            # 注册工具
            self._register_tools_for_role(role, state)
            
            # 执行原节点函数 (尝试智能透传参数)
            # 这里的逻辑是：如果 LangGraph 传了 config，我们尝试传给用户的函数
            # 如果用户的函数只接受 state，我们捕获错误并降级调用
            try:
                if asyncio.iscoroutinefunction(node_func):
                    return await node_func(state, config)
                else:
                    return node_func(state, config)
            except TypeError as e:
                # 如果是因为参数过多导致的错误，尝试只传 state
                # 注意：这里需要小心区分是“函数内部的TypeError”还是“参数匹配的TypeError"
                # 简单起见，我们假设用户的大部分节点只接受 state
                if "argument" in str(e): 
                    if asyncio.iscoroutinefunction(node_func):
                        return await node_func(state)
                    else:
                        return node_func(state)
                raise e # 抛出真正的逻辑错误
            finally:
                if token is not None:
                    try:
                        var_child_runnable_config.reset(token)
                    except Exception:
                        pass
        
        return wrapped_node

    def build(self, 
              checkpointer: Optional[BaseCheckpointSaver] = None, 
              **kwargs : Any):
        """
        重写父类的build方法。
        支持接收任意参数如 interrupt、checkpointer等
        """
        sg = StateGraph(self.state_model)
        
        # 添加节点（自动包装工具注册逻辑）
        for name, (func, role) in self.nodes.items():
            wrapped_func = self._wrap_node_with_tools(func, role)
            sg.add_node(name, wrapped_func)
        
        # 添加普通边
        for src, dst in self.edges:
            sg.add_edge(src, dst)
            
        # 添加条件边
        for src, cond_func in self.conditional_edges.items():
            sg.add_conditional_edges(src, cond_func)
            
        sg.set_entry_point(self.entry_point)
        return sg.compile(checkpointer=checkpointer, **kwargs)
        