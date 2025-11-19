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

from logger import AgentJSONLLogger


class Agent:
    def __init__(
        self,
        prompt_path: os.PathLike,
        model_name: str = "gpt-5",
        temperature: float = 0.8,
        retriever=None,
        tools=None,
        logger: Optional[AgentJSONLLogger] = None,
    ):
        self.prompt_path = prompt_path
        self.tools = tools or []
        self.retriever = retriever
        self.chain = None
        self.logger = logger
        self._last_retrieved_docs: Optional[List[Any]] = None

        # Add tools to the LLM
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

        # Setup the agent
        self.load_system_prompt()
        self.build_prompt_template()

        if not self.chain:
            if self.retriever is not None:
                self.build_retrieval_chain()
            else:
                self.build_chain()

    def load_system_prompt(self):
        try:
            with open(self.prompt_path, "r") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            print(f"[Error: System prompt file: {self.prompt_path} not found, agent not specialized.]")
            self.system_prompt = ""
            return

    def build_prompt_template(self):
        prompt = self.system_prompt

        if self.retriever: prompt += "\n\nContext: {context}"

        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

    def build_retrieval_chain(self):
        self.chain = (
            RunnableParallel({
                "context": lambda x: self.format_docs(
                    self.retriever.invoke(x["messages"][-1].content)
                ),
                "messages": lambda x: x["messages"]
            })
            | self.prompt_template
            | self.llm
        )

    def build_chain(self):
        self.chain = self.prompt_template | self.llm

    def format_docs(self, docs: list) -> str:
        if self.logger is not None:
            self._last_retrieved_docs = [
                {
                    "content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {}),
                }
                for doc in docs
            ]
        return "\n\n".join([doc.page_content for doc in docs])

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

        diagnostics = {
            "prompt_path": self.prompt_path,
            "uses_retriever": self.retriever is not None,
        }
        if self._last_retrieved_docs is not None:
            diagnostics["retrieved_documents"] = self._last_retrieved_docs
            self._last_retrieved_docs = None

        self.logger.log(
            {
                "agent_name": getattr(self, "name", self.prompt_path),
                "event": "agent_invocation",
                "input_messages": serialized_input,
                "output_messages": serialized_output,
                "tool_calls": tool_logs,
                "diagnostics": diagnostics,
            }
        )

    def __call__(self, state):
        print("="*60)
        print("Agent called: ", self.prompt_path)
        print("="*60)
        print("")
        messages = state.get("messages", [])

        result = self.chain.invoke({"messages": messages})
        tool_logs = []

        # Case 1: Model called a tool
        if hasattr(result, "tool_calls") and result.tool_calls:
            tool_messages = []
            for call in result.tool_calls:
                tool_name = call["name"]
                tool_input = call["args"]

                print("-"*60)
                print("Tool called: ", call["name"],"()")
                print("-"*60)

                matching_tool = next((t for t in self.tools if t.name == tool_name), None)
                if not matching_tool:
                    raise ValueError(f"Tool '{tool_name}' not found in agent's tool list.")

                # Execute the tool and capture output
                tool_output = matching_tool.invoke(tool_input)
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=str(tool_output)
                    )
                )
                tool_logs.append(
                    {
                        "tool_name": tool_name,
                        "args": tool_input,
                        "output": str(tool_output),
                    }
                )

            response_messages = [result] + tool_messages
            self._log_invocation(
                input_messages=messages,
                output_messages=response_messages,
                tool_logs=tool_logs,
            )
            return {"messages": response_messages}

        # Case 2: No tool call
        self._log_invocation(
            input_messages=messages,
            output_messages=[result],
            tool_logs=tool_logs,
        )
        return {"messages": [result]}
