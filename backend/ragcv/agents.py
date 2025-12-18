# This file will contain the various agents that will be used in the graph
import os
from typing import Any, List, Optional, Dict, Tuple, Sequence

from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from .utils.logger import JSONLLogger

class Agent:
    def __init__(
        self,
        name: str,
        prompt_path: os.PathLike,
        system_prompt_path: os.PathLike = "prompts/sys/system.txt",
        model_name: str = "gpt-5.1",
        temperature: float = 0.8,
        tools: Optional[List[Any]] = None,
        top_p: float = 1,
        logger: Optional[JSONLLogger] = None,
    ) -> None:  
        self.name = name
        self.prompt_path = prompt_path
        self.system_prompt_path = system_prompt_path
        self.tools = tools or []
        self.chain = None
        self.logger = logger

        # Add tools to the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
        )
        
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

        # Setup the agent
        self.system_prompt = ""
        self.load_system_prompt()
        self.build_prompt_template()
        self.build_chain()

    def load_system_prompt(self) -> None:
        for path in [self.system_prompt_path, self.prompt_path]:
            try:
                with open(path, "r") as f:
                    self.system_prompt += f.read()
                    
            except FileNotFoundError:
                print(f"[Error: System prompt file: {path} not found, {self.name} not specialized.]\n")
                return

    def build_prompt_template(self) -> None:
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

    def build_chain(self) -> None:
        self.chain = self.prompt_template | self.llm

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f'{"="*60}\nAgent called: {self.name}\n{"="*60}')

        messages = state.get("messages", [])
        result = self.chain.invoke({"messages": messages})
        tool_logs = []

        # Case 1: Model called a tool
        tool_messages = []
        if hasattr(result, "tool_calls") and result.tool_calls:
            tool_messages, tool_logs = process_tool_call(result, self.tools)
        
        # Add agent name to message
        agent_name = getattr(self, "name", self.prompt_path)

        if hasattr(result, "additional_kwargs") and isinstance(result.additional_kwargs, dict):
            result.additional_kwargs["agent_name"] = agent_name

        response_messages = [result] + tool_messages

        if self.logger is not None:     
            self.logger.log_agent_invocation(
                agent_name=agent_name,
                input_messages=messages,
                output_messages=response_messages,
                tool_logs=tool_logs,
            )
        return {"messages": response_messages}

def process_tool_call(
    result: Any,
    tools: Sequence[Any],
) -> Tuple[List[ToolMessage], List[Dict[str, Any]]]:
    tool_messages: List[ToolMessage] = []
    tool_logs: List[Dict[str, Any]] = []
    tool_map = {t.name: t for t in tools}

    for call in result.tool_calls:
        tool_name = call["name"]
        tool_input = call["args"]

        print(f'{"-"*60}\nTool called: {call["name"]}()\n{"-"*60}')

        matching_tool = tool_map.get(tool_name)
        if not matching_tool:
            raise ValueError(f"Tool '{tool_name}' not found in agent's tool list.")

        try:
            tool_output = matching_tool.invoke(tool_input)
        except Exception as e:
            tool_output = f"[Error executing tool: {e}]"

        tool_messages.append(
            ToolMessage(tool_call_id=call["id"], content=str(tool_output))
        )
        tool_logs.append(
            {"tool_name": tool_name, "args": tool_input, "output": str(tool_output)}
        )

    return tool_messages, tool_logs