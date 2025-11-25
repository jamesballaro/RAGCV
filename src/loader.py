# This file will parse the data in data/ and process it into a format that can be used by the pipeline (FAISS vector database)
import os
import re
import json
import tiktoken

from copy import deepcopy
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataLoader():
    def __init__(self, data_path: os.PathLike, db_path: os.PathLike, chunk_size: int = 1024, chunk_overlap: int = 256 ):
        self.data_path = data_path
        self.db_path = db_path
        self.loaders = None
        self.documents = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.schema_version = "retrieval_fidelity_v1"

        self.cv_folder_path = os.path.join(data_path, "CVs")
        self.cl_folder_path = os.path.join(data_path, "Cover Letters")
        self.notes_folder_path = os.path.join(data_path, "Notes")

        self.load()

    def load(self):
        # Load the data
        print("="*60)
        print("Loading data from files")
        print("="*60)
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

    @staticmethod
    def _sentence_tokenize(text: str) -> List[str]:
        cleaned = text.replace("\r", " ").strip()
        if not cleaned:
            return []
        # more robust splitter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', cleaned)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    @staticmethod
    def _chunk_sentences_token_budget(sentences: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = token_count(sent)
            if current and current_tokens + sent_tokens > max_tokens:
                chunks.append(" ".join(current).strip())

                # compute overlap sentences
                overlap = []
                overlap_acc = 0
                for s in reversed(current):
                    t = token_count(s)
                    if overlap_acc + t > overlap_tokens:
                        break
                    overlap.insert(0, s)
                    overlap_acc += t

                current = overlap.copy()
                current_tokens = sum(token_count(s) for s in current)

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current).strip())

        return [c for c in chunks if len(c) > 0]

    @staticmethod
    def _chunk_notes_sentences(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        sentences = DataLoader._sentence_tokenize(text)
        return DataLoader._chunk_sentences_token_budget(sentences, max_tokens, overlap_tokens)

    def _default_chunker(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    @staticmethod
    def _token_length(text: str) -> int:
        return token_count(text)

    @staticmethod
    def _split_bullets(text: str) -> List[str]:
        lines = text.splitlines()
        bullets = []
        current = []
        for ln in lines:
            if re.match(r'^(\s*[-•*]|\s*\d+\.)\s+', ln):
                if current:
                    bullets.append(" ".join(current).strip())
                current = [ln.strip()]
            else:
                if current:
                    current.append(ln.strip())
                else:
                    current = [ln.strip()]
        if current:
            bullets.append(" ".join(current).strip())
        return [b for b in bullets if len(b) > 0]

    def chunk_cv(self, text: str) -> List[str]:
        bullets = self._split_bullets(text)
        if not bullets:
            return self._default_chunker(text)

        max_tokens = 450
        overlap_tokens = 80

        chunks = []
        current = []
        current_tokens = 0

        for b in bullets:
            bt = token_count(b)
            if current and current_tokens + bt > max_tokens:
                chunks.append(" ".join(current).strip())

                # overlap last bullet
                overlap = []
                acc = 0
                for x in reversed(current):
                    tx = token_count(x)
                    if acc + tx > overlap_tokens:
                        break
                    overlap.insert(0, x)
                    acc += tx

                current = overlap.copy()
                current_tokens = sum(token_count(x) for x in current)

            current.append(b)
            current_tokens += bt

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    def chunk_cover_letter(self, text: str) -> List[str]:
        sentences = self._sentence_tokenize(text)
        return self._chunk_sentences_token_budget(
            sentences,
            max_tokens=450,
            overlap_tokens=120,
        )

    def chunk_notes(self, text: str) -> List[str]:
        return self._chunk_notes_sentences(
            text,
            max_tokens=400,
            overlap_tokens=120,
        )

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
        print("* Building vectorstore")
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
                    chunks = self.chunk_cv(content)
                elif doc_type == 'cl':
                    chunks = self.chunk_cover_letter(content)
                elif doc_type == 'notes':
                    chunks = self.chunk_notes(content)
                else:
                    chunks = self._default_chunker(content)

                metadata = deepcopy(doc.metadata)
                metadata["doc_type"] = doc_type
                doc_source = metadata.get("source")
                metadata["source"] = doc_source

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

        print("-- Success")

        return vectorstore

    def load_vectorstore(self, embeddings):
        if not self.db_similarity():
            vectorstore = self.build_vectorstore(embeddings=embeddings)
        else:
            print("\n [Corpus unchanged since last build, loading existing vectorstore.]\n")
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

def _get_encoding(encoding_name: Optional[str] = None):
    try:
        if encoding_name:
            return tiktoken.get_encoding(encoding_name)
        # Prefer model-based encoding if available; fall back to cl100k_base.
        try:
            return tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        # If tiktoken fails for any reason, raise an explicit error so caller
        raise RuntimeError("tiktoken initialisation failed") from e


def token_count(text: str, encoding=None) -> int:
    if not text:
        return 0
    enc = encoding or _get_encoding()
    return len(enc.encode(text))