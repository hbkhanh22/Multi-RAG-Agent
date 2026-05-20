"""
Helper Functions - Utility functions for the application
"""

from typing import List, Dict, Any, Optional
import time
import json
from datetime import datetime
from pathlib import Path

from src.utils.logger import app_logger


class Timer:
    """Context manager for measuring execution time"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        app_logger.info(f"{self.name} completed in {duration:.2f} seconds")
    
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0


def format_documents_for_display(documents: List, max_char: int = 500) -> List[Dict]:
    """
    Format documents for display
    
    Args:
        documents: List of Document objects
        max_char: Maximum characters per document
    
    Returns:
        List of formatted document dicts
    """
    formatted = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:max_char]
        if len(doc.page_content) > max_char:
            content += "..."
        
        formatted.append({
            "id": i,
            "source": doc.metadata.get("source", "Unknown"),
            "content": content,
            "metadata": doc.metadata
        })
    
    return formatted


def save_to_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indent level
    
    Returns:
        Success status
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        app_logger.info(f"Saved data to {filepath}")
        return True
    except Exception as e:
        app_logger.error(f"Error saving to {filepath}: {e}")
        return False


def load_from_json(filepath: str) -> Optional[Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data or None on error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        app_logger.info(f"Loaded data from {filepath}")
        return data
    except Exception as e:
        app_logger.error(f"Error loading from {filepath}: {e}")
        return None


def create_session_id() -> str:
    """Create a unique session ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_response(
    success: bool,
    data: Any = None,
    message: str = None,
    error: str = None,
    metadata: Dict = None
) -> Dict:
    """
    Format standard response
    
    Args:
        success: Success status
        data: Response data
        message: Optional message
        error: Optional error message
        metadata: Optional metadata
    
    Returns:
        Formatted response dict
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    if message:
        response["message"] = message
    
    if error:
        response["error"] = error
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_text_summary(text: str, num_sentences: int = 3) -> str:
    """Get first N sentences from text"""
    sentences = text.split(".")
    summary = ".".join(sentences[:num_sentences]).strip()
    if not summary.endswith("."):
        summary += "."
    return summary


def batch_items(items: List, batch_size: int) -> List[List]:
    """Batch items into chunks"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def get_nested_value(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Get nested value from dict"""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default


def validate_response_format(response: Dict) -> bool:
    """Validate response format"""
    required_fields = ["success"]
    return all(field in response for field in required_fields)


class CacheManager:
    """Simple in-memory cache manager"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


# Global cache instance
cache = CacheManager()
