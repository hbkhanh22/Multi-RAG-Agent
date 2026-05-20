from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int=512, chunk_overlap: int=128):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
    
    def chunk(self, documents: list):
        return self.text_splitter.split_documents(documents)
