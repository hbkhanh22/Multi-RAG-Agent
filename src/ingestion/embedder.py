from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional
from langchain_core.documents import Document

# Import settings
from config.settings import settings

class VectorStoreBuilder:
    def __init__(self, persist_path: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            output_dimensionality=settings.embedding_dimension
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
