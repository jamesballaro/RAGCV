# This file will parse the data in data/ and process it into a format that can be used by the pipeline (FAISS vector database)
import os
import re
import json
from copy import deepcopy
from typing import List, Dict, Iterable

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        #TODO: Add JSON parsing?
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
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    @staticmethod
    def _select_overlap(sentences: List[str], overlap_chars: int) -> List[str]:
        if overlap_chars <= 0:
            return []
        acc = 0
        overlap = []
        for sentence in reversed(sentences):
            overlap.insert(0, sentence)
            acc += len(sentence)
            if acc >= overlap_chars:
                break
        return overlap

    @staticmethod
    def _chunk_by_sentences(
        sentences: List[str],
        max_chars: int,
        overlap_chars: int,
    ) -> List[str]:
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # account for spacing
            if current and current_len + sentence_len > max_chars:
                chunks.append(" ".join(current).strip())
                overlap_seed = DataLoader._select_overlap(current, overlap_chars)
                current = overlap_seed.copy()
                current_len = sum(len(s) + 1 for s in current)

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(" ".join(current).strip())

        return [c for c in chunks if len(c) > 0]

    @staticmethod
    def _chunk_small(lines: Iterable[str], chunk_size: int, overlap: int) -> List[str]:
        buffer = []
        buffer_len = 0
        chunks: List[str] = []

        for line in lines:
            normalized = line.strip()
            if not normalized:
                continue
            line_len = len(normalized) + 1
            if buffer and buffer_len + line_len > chunk_size:
                chunks.append("\n".join(buffer))
                # overlap by characters
                overlap_text = "\n".join(buffer)
                keep_len = max(0, len(overlap_text) - overlap)
                buffer = [overlap_text[-keep_len:]] if keep_len else []
                buffer_len = len("\n".join(buffer))

            buffer.append(normalized)
            buffer_len += line_len

        if buffer:
            chunks.append("\n".join(buffer))

        return chunks

    def _default_chunker(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def chunk_cv(self, text: str) -> List[str]:
        paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 30]
        chunks: List[str] = []

        for paragraph in paragraphs:
            sentences = self._sentence_tokenize(paragraph)
            chunks.extend(
                self._chunk_by_sentences(
                    sentences,
                    max_chars=1500,
                    overlap_chars=256,
                )
            )

        return chunks or self._default_chunker(text)

    def chunk_cover_letter(self, text: str) -> List[str]:
        sentences = self._sentence_tokenize(text)
        chunks = self._chunk_by_sentences(
            sentences,
            max_chars=1200,
            overlap_chars=200,
        )
        return chunks or self._default_chunker(text)

    def chunk_notes(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks = self._chunk_small(
            lines,
            chunk_size=600,
            overlap=200,
        )
        return chunks or self._default_chunker(text)

    @staticmethod
    def _estimate_token_length(text: str) -> int:
        # Fallback heuristic: assume ~4 chars per token
        return max(1, round(len(text) / 4))

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
                            "chunk_length_tokens": self._estimate_token_length(chunk),
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