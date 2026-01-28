from typing import Any, List, Optional, Dict, Tuple, Sequence, Type
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Agent:
    def __init__(
        self,
        name: str,
        prompt: ChatPromptTemplate,
        output_parser: Type[BaseModel],

        model_name: str = "gpt-5-nano",
        temperature: float = 0.8,
        tools: Optional[List[Any]] = None,
        top_p: float = 1,
    ) -> None:  
        self.name = name
        self.prompt = prompt
        self.output_parser = output_parser

        self.tools = tools or []
        self.chain = None
        self.tool_chain = None

        # Add tools to the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            streaming=True
        )
        
        if self.tools:
            self.chain = (
                self.prompt
                | self.llm.bind_tools(self.tools)
                .with_structured_output(self.output_parser, method="function_calling")
            )
        else:
            self.chain = self.prompt | self.llm.with_structured_output(
                self.output_parser, method="function_calling"
            )

    def invoke(self, input_data: Dict[str, Any], callbacks = None) -> Dict[str, Any]:

        tool_result = None
        
        if self.tool_chain:
            tool_result = self.tool_chain.invoke(input_data)
            
        config = {"callbacks": [callbacks()]} if callbacks else {}
        result = self.chain.invoke(input_data, config=config)

        return result, tool_result
            
