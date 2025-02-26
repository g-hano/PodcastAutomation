"""Data models for podcast structure and content."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Speaker:
    """Represents a podcast speaker."""
    name: str
    voice_path: str


@dataclass
class Exchange:
    """Represents a single speech exchange in a conversation."""
    speaker: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    original_content: Optional[str] = None


@dataclass
class Topic:
    """Represents a podcast discussion topic with associated exchanges."""
    title: str
    exchanges: List[Exchange] = field(default_factory=list)


@dataclass
class Podcast:
    """Data model for a complete podcast."""
    title: str = ""
    description: str = ""
    intro: str = ""
    outro: str = ""
    topics: List[Topic] = field(default_factory=list)
    document_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Optional translated versions
    original_intro: Optional[str] = None
    original_outro: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert podcast to dictionary format for serialization."""
        return {
            "metadata": {
                "title": self.title,
                "description": self.description,
                "timestamp": self.created_at,
                "document": self.document_text,
                "total_topics": len(self.topics)
            },
            "intro": self.intro,
            "conversations": {
                topic.title: [exchange.__dict__ for exchange in topic.exchanges]
                for topic in self.topics
            },
            "outro": self.outro,
            **({"original_intro": self.original_intro} if self.original_intro else {}),
            **({"original_outro": self.original_outro} if self.original_outro else {})
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Podcast':
        """Create a podcast from a dictionary."""
        podcast = cls(
            title=data.get("metadata", {}).get("title", ""),
            description=data.get("metadata", {}).get("description", ""),
            intro=data.get("intro", ""),
            outro=data.get("outro", ""),
            document_text=data.get("metadata", {}).get("document", ""),
            created_at=data.get("metadata", {}).get("timestamp", datetime.now().isoformat()),
            original_intro=data.get("original_intro"),
            original_outro=data.get("original_outro")
        )
        
        # Add topics and exchanges
        for topic_title, exchanges_data in data.get("conversations", {}).items():
            topic = Topic(title=topic_title)
            for exchange_data in exchanges_data:
                exchange = Exchange(
                    speaker=exchange_data.get("speaker", ""),
                    content=exchange_data.get("content", ""),
                    timestamp=exchange_data.get("timestamp", ""),
                    original_content=exchange_data.get("original_content")
                )
                topic.exchanges.append(exchange)
            podcast.topics.append(topic)
        
        return podcast