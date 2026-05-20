from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from pathlib import Path

class DocumentLoader:
    def load_from_directory(self, path: str, glob: str = "**/*.pdf"):
        loader = DirectoryLoader(
            path, glob=glob,
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return loader.load()