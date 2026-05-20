"""
Vector Store Module - Manages Chroma vector databases for each domain
Using Google's embedding models
"""

import os
from typing import List, Optional
from pathlib import Path
import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector stores for different domains"""
    
    def __init__(self):
        """Initialize vector store manager with Google embeddings"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.google_api_key,
                transport="rest"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
        
        self.stores = {}
        self._load_or_create_stores()
    
    def _load_or_create_stores(self):
        """Load existing stores or create new ones"""
        store_paths = {
            "finance": settings.finance_store_path,
            "tech": settings.tech_store_path,
            "general": settings.general_store_path,
        }
        
        for domain, path in store_paths.items():
            try:
                # Check if store exists
                if os.path.exists(path):
                    logger.info(f"Loading existing {domain} vector store from {path}")
                    self.stores[domain] = Chroma(
                        persist_directory=str(path),
                        embedding_function=self.embeddings,
                        collection_name=f"{domain}_collection"
                    )
                else:
                    logger.info(f"Creating new {domain} vector store at {path}")
                    Path(path).mkdir(parents=True, exist_ok=True)
                    # Create empty store - will be populated later
                    self.stores[domain] = Chroma(
                        persist_directory=str(path),
                        embedding_function=self.embeddings,
                        collection_name=f"{domain}_collection"
                    )
            except Exception as e:
                logger.error(f"Error loading/creating {domain} store: {e}")
                self.stores[domain] = None
    
    def add_documents(self, domain: str, documents: List[Document]) -> bool:
        """
        Add documents to a domain's vector store
        
        Args:
            domain: Domain name (finance, tech, general)
            documents: List of Document objects
        
        Returns:
            Success status
        """
        try:
            if domain not in self.stores or self.stores[domain] is None:
                logger.error(f"Store for domain '{domain}' not initialized")
                return False
            
            logger.info(f"Adding {len(documents)} documents to {domain} store")
            self.stores[domain].add_documents(documents)
            self.stores[domain].persist()
            
            logger.info(f"Successfully added documents to {domain} store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {domain} store: {e}")
            return False
    
    def get_retriever(self, domain: str, k: int = None, search_type: str = "mmr"):
        """
        Get a retriever for a specific domain
        
        Args:
            domain: Domain name
            k: Number of results (uses top_k_retrieval from settings if not specified)
            search_type: "similarity" or "mmr" (maximum marginal relevance)
        
        Returns:
            LangChain retriever object
        """
        if k is None:
            k = settings.top_k_retrieval
        
        if domain not in self.stores or self.stores[domain] is None:
            logger.error(f"Store for domain '{domain}' not available")
            return None
        
        try:
            retriever = self.stores[domain].as_retriever(
                search_type=search_type,
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 2  # Fetch more to rerank
                }
            )
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever for {domain}: {e}")
            return None
    
    def search(self, domain: str, query: str, k: int = None) -> List[Document]:
        """
        Search for documents in a domain
        
        Args:
            domain: Domain name
            query: Search query
            k: Number of results
        
        Returns:
            List of relevant documents
        """
        if k is None:
            k = settings.top_k_retrieval
        
        try:
            if domain not in self.stores or self.stores[domain] is None:
                logger.error(f"Store for domain '{domain}' not available")
                return []
            
            results = self.stores[domain].similarity_search_with_score(
                query, k=k
            )
            
            # Filter by minimum relevance score
            min_score = 0.1  # Adjust based on your needs
            filtered_results = [
                doc for doc, score in results if score >= min_score
            ]
            
            logger.info(f"Found {len(filtered_results)} documents for query in {domain}")
            return [doc[0] for doc in results]  # Return just documents
        except Exception as e:
            logger.error(f"Error searching {domain} store: {e}")
            return []
    
    def delete_documents(self, domain: str, document_ids: List[str]) -> bool:
        """Delete specific documents from a store"""
        try:
            if domain not in self.stores or self.stores[domain] is None:
                return False
            
            for doc_id in document_ids:
                self.stores[domain].delete([doc_id])
            
            self.stores[domain].persist()
            return True
        except Exception as e:
            logger.error(f"Error deleting documents from {domain}: {e}")
            return False
    
    def get_store_stats(self, domain: str) -> dict:
        """Get statistics about a vector store"""
        try:
            if domain not in self.stores or self.stores[domain] is None:
                return {"error": "Store not available"}
            
            collection = self.stores[domain]._collection
            count = collection.count()
            
            return {
                "domain": domain,
                "document_count": count,
                "embedding_dimension": settings.embedding_dimension,
                "embedding_model": settings.embedding_model,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting stats for {domain}: {e}")
            return {"error": str(e)}
    
    def clear_store(self, domain: str) -> bool:
        """Clear all documents from a store"""
        try:
            if domain not in self.stores or self.stores[domain] is None:
                return False
            
            # Delete and recreate
            path = self.stores[domain]._persist_directory
            import shutil
            if os.path.exists(path):
                shutil.rmtree(path)
            
            Path(path).mkdir(parents=True, exist_ok=True)
            self.stores[domain] = Chroma(
                persist_directory=path,
                embedding_function=self.embeddings,
                collection_name=f"{domain}_collection"
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing {domain} store: {e}")
            return False


# Singleton instance
vector_store_manager = VectorStoreManager()
