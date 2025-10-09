# This file will parse the data in data/ and process it into a format that can be used by the pipeline (FAISS vector database)
import os
import operator
import re
import json

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

def compile_docs(loaders, folder):
    for loader in loaders:
        doc = loader.load()
        folder.extend(doc)
        if isinstance(doc, list) and all(isinstance(d, Document) for d in doc):
            print("\t * Loaded doc from:", doc[0].metadata['source'])
        else:
            print("\t * Error loading:", doc[0].metadata['source'])


class DataLoader():
    def __init__(self, data_path: os.PathLike, db_path: os.PathLike, chunk_size: int = 1024, chunk_overlap: int = 256 ):
        self.data_path = data_path
        self.db_path = db_path
        self.loaders = None
        self.documents = None #TODO: Implement struct
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.cv_folder_path = os.path.join(data_path, "CVs")
        self.cl_folder_path = os.path.join(data_path, "coverletters")
        self.tp_folder_path = os.path.join(data_path, "templates")

        self.load()

    def load(self):
        #TODO: Add JSON parsing?
        if self.loaders is None:
            cv_loaders = [self.create_loader(pdf, self.cv_folder_path) for pdf in os.listdir(self.cv_folder_path)]
            cl_loaders = [self.create_loader(pdf, self.cl_folder_path) for pdf in os.listdir(self.cl_folder_path)]
            tp_loaders = [self.create_loader(pdf, self.tp_folder_path) for pdf in os.listdir(self.tp_folder_path)]

            if self.documents is None:
                self.all_cvs = []
                self.all_cls = []
                self.all_tps = []

            self.documents = [
                {
                    "type":"cv",
                    "loaders":cv_loaders,
                    "folder":self.all_cvs
                },
                {
                    "type":"cl",
                    "loaders":cl_loaders,
                    "folder":self.all_cls
                },
                {
                    "type":"tp",
                    "loaders":tp_loaders,
                    "folder":self.all_tps
                }
            ]

        for cat in self.documents:
              compile_docs(cat['loaders'], cat['folder'])

        self.all_files = [doc.metadata['source'] for doc in self.all_cvs + self.all_cls + self.all_tps]
        return

    def create_loader(self, filename, folder):
        full_path = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            try:
                return PyPDFLoader(full_path)
            except Exception:
                return UnstructuredPDFLoader(full_path)
        elif ext == ".tex" or ext == ".txt":
            return TextLoader(full_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def chunk_cv(self,text: str) -> List[str]:
        # Split into sections by uppercase headers
        sections = re.split(r"\n(?=[A-Z][A-Z ]{2,})", text)
        bullets = []
        for section in sections:
            sub = re.split(r"[\n•\-]\s+", section.strip())
            bullets.extend([s.strip() for s in sub if len(s.strip()) > 30])

        # Re-chunk for uniform embedding size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n", ".", " "],
        )
        final_chunks = splitter.split_text("\n".join(bullets))
        return final_chunks


    def chunk_cover_letter(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        joined = "\n\n".join(paragraphs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", ".", " "],
        )
        final_chunks = splitter.split_text(joined)
        return final_chunks

    def chunk_template(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        final_chunks = splitter.split_text(text)
        return final_chunks

    def generate_db_metadata(self):
        metadata = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "data_path": self.data_path,
            "db_path": self.db_path,
            "doc_list": self.all_files
        }
        return metadata

    def db_similarity(self):
        current_metadata = self.generate_db_metadata()
        try:
            with open(os.path.join(self.db_path, "metadata.json"), "r", encoding="utf-8") as f:
                previous_metadata = json.load(f)
            for key, value in current_metadata.items():
                if key not in previous_metadata:
                    return False
                elif current_metadata[key] != previous_metadata[key]:
                    return False

        except FileNotFoundError:
            return False

        return True


    def build_vectorstore(self):
        print("* Building vectorstore")
        # Create embeddings
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        all_docs = []

        # Go through each document type in self.documents, chunking, and adding to the vectorstore
        for docs in self.documents:
            doc_type = docs['type']

            for doc in docs['folder']:
                content = doc.page_content
                if doc_type == 'cv':
                    chunks = self.chunk_cv(content)
                elif doc_type == 'cl':
                    chunks = self.chunk_cover_letter(content)
                else:
                    chunks = self.chunk_template(content)

                metadata = doc.metadata
                # Attach metadata to each chunk
                doc_chunks = [
                    Document(page_content=chunk, metadata=metadata)
                    for chunk in chunks
                ]
                all_docs.extend(doc_chunks)

        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(self.db_path)

        # Store db metadata
        metadata_path = os.path.join(self.db_path, "metadata.json")
        db_metadata = self.generate_db_metadata()

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(db_metadata, f, indent=4)

        print("-- Success")

        return vectorstore


