# This file will contain the various agents that will be used in the graph
import os
import operator

from typing import TypedDict, List, Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

# Create an agent class to easily create specialized agents
class Agent():
    def __init__(self,
            prompt_path: os.PathLike,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.8,
            retriever = None,
            tools = None,
            ):
        self.prompt_path = prompt_path
        self.tools = tools or []
        self.retriever = retriever
        self.chain = None

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

    def build_prompt_template(self, human_template="{input}"):
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

    #TODO: should this be here?
    def format_docs(self, docs: list) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    def __call__(self, state):
        print("="*60)
        print("Agent called: ", self.prompt_path)
        print("="*60)
        print("")
        messages = state.get("messages", [])
        
        result = self.chain.invoke({"messages": messages})

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

            return {"messages": [result] + tool_messages}

        # Case 2: No tool call
        return {"messages": [result]}
