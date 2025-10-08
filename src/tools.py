import os
import operator

from typing import TypedDict, List, Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

@tool("write_to_file", return_direct=True)
def write_to_file(content: str, filename: str = "output.txt") -> str:
    """
    Writes text content to a file within the ./output directory.
    """
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"[FileTool] Output successfully written to {filepath}"
