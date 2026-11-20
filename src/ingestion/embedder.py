"""
Stand-alone Chroma builder (notebooks / scripts). The application itself indexes
through VectorStoreManager. Heavy imports are deferred so importing `src.ingestion`
does not require chromadb or torch (the ONNX runtime image ships neither).
"""

from typing import List

from langchain_core.documents import Document

from config.settings import settings


class VectorStoreBuilder:
    def __init__(self, persist_path: str):
        from src.retrieval.vector_store import make_embeddings

        self.embeddings = make_embeddings()
        self.persist_path = persist_path

    def build(self, documents: List[Document]):
        from langchain_chroma import Chroma

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_path,
        )

    def load(self):
        from langchain_chroma import Chroma

        return Chroma(persist_directory=self.persist_path, embedding_function=self.embeddings)
