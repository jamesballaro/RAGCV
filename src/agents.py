# This file will contain the various agents that will be used in the graph
import os
from typing import Any, List, Optional

from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from .logger import JSONLLogger

class Agent:
    def __init__(
        self,
        name,
        prompt_path: os.PathLike,
        system_prompt_path="prompts/system.txt",
        model_name: str = "gpt-5",
        temperature: float = 0.8,
        tools = None,
        top_p = 1,
        logger: Optional[JSONLLogger] = None,
    ):  
        self.name = name
        self.prompt_path = prompt_path
        self.system_prompt_path=system_prompt_path
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
        self.system_prompt=""
        self.load_system_prompt()
        self.build_prompt_template()
        self.build_chain()

    def load_system_prompt(self):
        for path in [self.system_prompt_path, self.prompt_path]:
            try:
                with open(path, "r") as f:
                    self.system_prompt += f.read()
            except FileNotFoundError:
                print(f"[Error: System prompt file: {self.prompt_path} not found, agent not specialized.]")
                return

    def build_prompt_template(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

    def build_chain(self):
        self.chain = self.prompt_template | self.llm

    def _serialize_message(self, message: BaseMessage) -> dict:
        return {
            "type": message.__class__.__name__,
            "content": getattr(message, "content", None),
            "name": getattr(message, "name", None),
            "tool_calls": getattr(message, "tool_calls", []),
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
            "response_metadata": getattr(message, "response_metadata", {}),
        }

    def _log_invocation(
        self,
        *,
        input_messages: List[AnyMessage],
        output_messages: List[AnyMessage],
        tool_logs: List[dict],
    ) -> None:
        if self.logger is None:
            return

        serialized_input = [self._serialize_message(msg) for msg in input_messages]
        serialized_output = [self._serialize_message(msg) for msg in output_messages]
        
        self.logger.log(
            {
                "agent_name": getattr(self, "name", self.prompt_path),
                "event": "agent_invocation",
                "input_messages": serialized_input,
                "output_messages": serialized_output,
                "tool_calls": tool_logs,
            }
        )

    def __call__(self, state):
        print("="*60, "\nAgent called: ", self.prompt_path)
        print("="*60)

        messages = state.get("messages", [])
        result = self.chain.invoke({"messages": messages})
        tool_logs = []

        # Case 1: Model called a tool
        tool_messages = []
        if hasattr(result, "tool_calls") and result.tool_calls:
            tool_messages, tool_logs = process_tool_call(result, self.tools)
        
        response_messages = [result] + tool_messages

        # Case 2: No tool call
        self._log_invocation(
            input_messages=messages,
            output_messages=response_messages,
            tool_logs=tool_logs,
        )
        return {"messages": response_messages}

def process_tool_call(result, tools):
    tool_messages = []
    tool_logs = []
    tool_map = {t.name: t for t in tools}

    for call in result.tool_calls:
        tool_name = call["name"]
        tool_input = call["args"]

        print("-"*60,"\nTool called: ", call["name"],"()\n","-"*60)

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