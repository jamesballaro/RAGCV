# This file will parse the data in data/ and process it into a format that can be used by the pipeline (FAISS vector database)
import os
import pydf

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

class PDFLoader():
    def __init__(self, data_path: os.PathLike, db_path: os.PathLike, chunk_size: int = 1024, overlap: int = 256 ):
        self.data_path = data_path
        self.db_path = db_path
        self.loaders = None
        self.documents = None #TODO: Implement struct
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.cv_folder_path = os.path.join(data_path, "CVs")
        self.cl_folder_path = os.path.join(data_path, "coverletters")

    def load(self):
        #TODO: Add JSON parsing?
        print("* Loading data")
        if self.loaders is None:
            cv_loaders = [create_loader(pdf) for pdf in os.listdir(cv_folder_path)]
            cl_loaders = [create_loader(pdf) for pdf in os.listdir(cl_folder_path)]
            self.loaders = [cv_loaders, cl_loaders]

        if self.documents is None:
           self.all_cvs = []
           self.all_cls = []

           for cv_doc, cl_doc in self.loaders.load():
               self.all_cvs = all_cvs + cv_doc
               self.all_cls = all_cls + cl_doc

        self.documents = [
            {
                'type': 'cv',
                'docs': self.all_cvs
            },
            {
                'type': 'cl',
                'docs': self.all_cls
            }
        ]
        print("-- Success")
        return

    def create_loader(path):
        # PyPDFLoader is faster but UnstructuredPDFLoader can handle images, so we handle both
        try:
            return PyPDFLoader(path)
        except Exception:
            return UnstructuredPDFLoader(path)

    def chunk_cv(self,text: str) -> List[str]:
        print("* Chunking CV")
        # Split into sections by uppercase headers
        sections = re.split(r"\n(?=[A-Z][A-Z ]{2,})", text)
        bullets = []
        for section in sections:
            sub = re.split(r"[\n•\-]\s+", section.strip())
            bullets.extend([s.strip() for s in sub if len(s.strip()) > 30])

        # Re-chunk for uniform embedding size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n", ".", " "],
        )
        final_chunks = splitter.split_text("\n".join(bullets))
        print("-- Success")
        return final_chunks


    def chunk_cover_letter(self, text: str) -> List[str]:
        print("* Chunking cover letter")
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        joined = "\n\n".join(paragraphs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", ".", " "],
        )
        final_chunks = splitter.split_text(joined)
        print("-- Success")
        return final_chunks

    def build_vectorstore(self):
        print("* Building vectorstore")
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        all_docs = []

        # Go through each document type in self.documents, chunking, and adding to the vectorstore
        for docs in self.documents:
            doc_type = docs['type']

            for doc in docs['docs']:
                if doc_type == 'cv':
                    chunks = self.chunk_cv(doc)
                elif doc_type == 'cl':
                    chunks = self.chunk_cover_letter(doc)

                print(doc.metadata)

                metadata = doc.metadata
                # Attach metadata to each chunk
                doc_chunk = [
                    {
                        "content": chunk,
                        "metadata": metadata,
                    }
                    for i, chunk in enumerate(chunks)
                ]
                all_docs.extend(doc_chunk)

        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(self.db_path)
        print("-- Success")

        return vectorstore


