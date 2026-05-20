from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
from pathlib import Path

from config.settings import settings
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import VectorStoreBuilder
from src.ingestion.loader import DocumentLoader

class HydridRetriver:
    def __init__(self, vectorstore: Chroma, docs: List[Document], top_k: int = 10):
        # Vector retrieer
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 2}
        )

        # BM25 Retriever
        bm25_retriever= BM25Retriever.from_documents(docs)
        bm25_retriever.k = top_k

        # Ensemble Retriever
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )

    def retrieve(self, query: str):
        """Retrieve documents using hybrid retriever"""
        return self.retriever.invoke(query)

def get_finance_retriever() -> HydridRetriver:
    vectorstore_builder = VectorStoreBuilder(settings.finance_store_path)
    vectorstore = vectorstore_builder.load()
    docs = _documents_from_vectorstore(vectorstore)

    if not docs:
        docs = _load_finance_documents()
        if not docs:
            raise ValueError(
                "No finance documents found. Add finance PDFs to data/raw/finance "
                "or build a Chroma store at settings.finance_store_path."
            )
        vectorstore = vectorstore_builder.build(docs)

    return HydridRetriver(
        vectorstore=vectorstore,
        docs=docs,
        top_k=settings.top_k_retrieval,
    )

def get_tech_retriever() -> HydridRetriver:
    vectorstore_builder = VectorStoreBuilder(settings.tech_store_path)
    vectorstore = vectorstore_builder.load()
    docs = _documents_from_vectorstore(vectorstore)

    if not docs:
        docs = _load_tech_documents()
        if not docs:
            raise ValueError(
                "No tech documents found. Add tech PDFs to data/raw/tech "
                "or build a Chroma store at settings.tech_store_path."
            )
        vectorstore = vectorstore_builder.build(docs)

    return HydridRetriver(
        vectorstore=vectorstore,
        docs=docs,
        top_k=settings.top_k_retrieval,
    )

def get_general_retriever() -> HydridRetriver:
    vectorstore_builder = VectorStoreBuilder(settings.general_store_path)
    vectorstore = vectorstore_builder.load()
    docs = _documents_from_vectorstore(vectorstore)

    if not docs:
        docs = _load_general_documents()
        if not docs:
            raise ValueError(
                "No general documents found. Add general PDFs to data/raw/general "
                "or build a Chroma store at settings.general_store_path."
            )
        vectorstore = vectorstore_builder.build(docs)

    return HydridRetriver(
        vectorstore=vectorstore,
        docs=docs,
        top_k=settings.top_k_retrieval,
    )

def _documents_from_vectorstore(vectorstore: Chroma) -> List[Document]:
    stored = vectorstore.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(documents, metadatas)
        if content
    ]


def _load_finance_documents() -> List[Document]:
    finance_path = Path("data/raw/finance")
    if not finance_path.exists():
        return []

    documents = DocumentLoader().load_from_directory(str(finance_path))
    return DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).chunk(documents)

def _load_tech_documents() -> List[Document]:
    tech_path = Path("data/raw/tech")
    if not tech_path.exists():
        return []

    documents = DocumentLoader().load_from_directory(str(tech_path))
    return DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).chunk(documents)

def _load_general_documents() -> List[Document]:
    general_path = Path("data/raw/general")
    if not general_path.exists():
        return []

    documents = DocumentLoader().load_from_directory(str(general_path))
    return DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).chunk(documents)