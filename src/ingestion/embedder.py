from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional
from langchain_core.documents import Document

# Import settings
from config.settings import settings

class VectorStoreBuilder:
    def __init__(self, persist_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.persist_path = persist_path

    def build(self, documents: List[Document]) -> Chroma:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_path
        )
        return vectorstore
    
    def load(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_path,
            embedding_function=self.embeddings
        )
