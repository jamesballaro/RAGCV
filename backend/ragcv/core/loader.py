# This file will parse the data in data/ and process it into a format that can be used by the pipeline (FAISS vector database)
import os
import re
import json

from copy import deepcopy
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from ..retrieval.chunking import (
    token_count,
    chunk_cover_letter, 
    chunk_cv, 
    chunk_notes, 
    default_chunker
)

class DataLoader():
    def __init__(
        self, 
        data_path: os.PathLike, 
        db_path: os.PathLike, 
        chunk_size: int = 1024, 
        chunk_overlap: int = 256 
    ):
        self.data_path = data_path
        self.db_path = db_path
        self.loaders = None
        self.documents = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.schema_version = "retrieval_fidelity_v1"

        self.cv_folder_path = os.path.join(data_path, "CVs")
        self.cl_folder_path = os.path.join(data_path, "CoverLetters")
        self.notes_folder_path = os.path.join(data_path, "Notes")

        self.load()

    def load(self):
        # Load the data
        num_files = 0
        for folder in [self.cv_folder_path, self.cl_folder_path, self.notes_folder_path]:
            if os.path.exists(folder):
                num_files += len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        print(f"{'='*60}\nLoading data from {num_files} files\n{'='*60}")
        if self.loaders is None:
            cv_loaders = [self.create_loader(f, self.cv_folder_path) for f in os.listdir(self.cv_folder_path)]
            cl_loaders = [self.create_loader(f, self.cl_folder_path) for f in os.listdir(self.cl_folder_path)]
            notes_loaders = [self.create_loader(f, self.notes_folder_path) for f in os.listdir(self.notes_folder_path)]

            if self.documents is None:
                self.all_cvs = []
                self.all_cls = []
                self.all_notes = []

            self.documents = [
                {
                    "type": "cv",
                    "loaders": cv_loaders,
                    "folder": self.all_cvs,
                },
                {
                    "type": "cl",
                    "loaders": cl_loaders,
                    "folder": self.all_cls,
                },
                {
                    "type": "notes",
                    "loaders": notes_loaders,
                    "folder": self.all_notes,
                }
            ]

        for cat in self.documents:
              compile_docs(cat["loaders"], cat["folder"], cat["type"])

        self.all_files = [doc.metadata["source"] for doc in self.all_cvs + self.all_cls + self.all_notes]
        return

    def get_documents(self):
        all_content = [doc for doc in self.all_cvs + self.all_cls + self.all_notes]
        return all_content

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

    def generate_db_metadata(self):
        metadata = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "data_path": self.data_path,
            "db_path": self.db_path,
            "doc_list": self.all_files,
            "schema_version": self.schema_version,
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

    def build_vectorstore(self, embeddings: OpenAIEmbeddings | None = None):
        print(f"{'='*60}\nBuilding Vectorstore")

        # Create embeddings
        if embeddings is None:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        all_docs = []

        # Go through each document type in self.documents, chunking, and adding to the vectorstore
        for docs in self.documents:
            doc_type = docs['type']

            for doc in docs['folder']:
                content = doc.page_content
                if doc_type == 'cv':
                    chunks = chunk_cv(content)
                elif doc_type == 'cl':
                    chunks = chunk_cover_letter(content)
                elif doc_type == 'notes':
                    chunks = chunk_notes(content)
                else:
                    chunks = default_chunker(content)

                metadata = deepcopy(doc.metadata)
                metadata["doc_type"] = doc_type
                doc_source = metadata.get("source")
                metadata["source"] = doc_source

                doc_chunks = []
                # Attach metadata to each chunk
                doc_chunks = [
                    Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            "chunk_length_tokens": token_count(chunk),
                        }
                    )
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

        print("-- Success\n","="*60)

        return vectorstore

    def load_vectorstore(self, embeddings):
        if not self.db_similarity():
            vectorstore = self.build_vectorstore(embeddings=embeddings)
        else:
            print("\n[Corpus unchanged since last build, loading existing vectorstore.]\n")
            vectorstore = FAISS.load_local(
                    self.db_path,
                    embeddings,
                    allow_dangerous_deserialization=True
            )
        return vectorstore
    
def compile_docs(loaders, folder: List[Document], doc_type: str):
    for loader in loaders:
        documents = loader.load()
        for doc in documents:
            doc.metadata = dict(doc.metadata)
            doc.metadata["doc_type"] = doc_type
        folder.extend(documents)
        if isinstance(documents, list) and all(isinstance(d, Document) for d in documents):
            print("\t * Loaded doc from:", documents[0].metadata["source"])
        else:
            print("\t * Error loading:", documents[0].metadata["source"])

