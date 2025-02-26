"""Content generation for podcast creation."""

import logging
from typing import List

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine

from ..core.podcast import Podcast, Topic, Exchange
from ..core.config import PodcastConfig
from ..utils.text_processing import clean_script_text

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generates podcast content including topics, intro, outro, and conversations."""
    
    # Language code to language name mapping
    LANGUAGE_NAMES = {
        'a': 'American English',
        'b': 'British English',
        'j': 'Japanese',
        'h': 'Hindi',
        'p': 'Portuguese',
        'z': 'Chinese',
        'i': 'Italian',
        'f': 'French',
        'e': 'Spanish'
    }
    
    def __init__(self, config: PodcastConfig):
        """Initialize the generator with configuration."""
        self.config = config
        self.num_topics = config.num_topics
        self.num_turns = config.num_turns
        self.target_lang_code = config.audio.lang
        self.target_language = self.LANGUAGE_NAMES.get(self.target_lang_code, 'American English')
        
        # Content is always generated in English first, then translated if needed
        logger.info(f"Content will be generated in English, target language is {self.target_language}")
        
        # Initialize LLM engines
        self.topic_engine = self._create_topic_engine()
        self.intro_engine = self._create_intro_engine()
        self.outro_engine = self._create_outro_engine()
        self.moderator_engine = self._create_moderator_engine()
        self.guest_engine = self._create_guest_engine()
        self.metadata_engine = self._create_metadata_engine()
    
    def _create_topic_engine(self) -> SimpleChatEngine:
        """Create the topic generation chat engine."""
        llm = Ollama(model=self.config.models.topic_generator, context_window=4096)
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system", 
                    content=f"""You are a creative podcast producer. Given a document about technology:
                    1. Use English (content will be translated later if needed)
                    2. Generate exactly {self.num_topics} specific, focused discussion topics
                    3. Topics should encourage debate and different perspectives
                    4. Each topic should be concrete and specific, not general
                    5. Focus on practical implications, technical innovations, and real-world impact
                    6. Each topic should be 10-15 words maximum"""
                )
            ]
        )
    
    def _create_intro_engine(self) -> SimpleChatEngine:
        """Create the introduction generation chat engine."""
        llm = Ollama(model=self.config.models.intro_generator, context_window=4096)
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content="""You are an engaging podcast host creating episode introductions. Your intro should:
                    1. Be in English (content will be translated later if needed)
                    2. Warmly greet the listeners
                    3. Create excitement about today's topics
                    4. Briefly preview each topic (1 sentence per topic)
                    5. Use casual, friendly language
                    6. Keep the total intro under 2 minutes when spoken
                    7. End with a transition to the first topic"""
                )
            ]
        )
    
    def _create_outro_engine(self) -> SimpleChatEngine:
        """Create the outro generation chat engine."""
        llm = Ollama(model=self.config.models.outro_generator, context_window=4096)
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content="""You are a podcast host creating episode conclusions. Your outro should:
                    1. Be in English (content will be translated later if needed)
                    2. Summarize key points from each topic (1-2 sentences each)
                    3. Thank listeners for tuning in
                    4. Encourage engagement (comments, feedback, suggestions)
                    5. Mention looking forward to the next episode
                    6. Keep it concise and friendly
                    7. Use natural, conversational language"""
                )
            ]
        )
    
    def _create_moderator_engine(self) -> SimpleChatEngine:
        """Create the moderator chat engine."""
        llm = Ollama(model=self.config.models.podcast_moderator, context_window=4096)
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content="""You're hosting a casual podcast. Keep it natural and flowing:
- Use English (content will be translated later if needed)
- Use 10-15 words max per response
- Ask ONE engaging question per turn
- Use conversational language like "Hey", "So", "Wow"
- React naturally to previous answers
- NEVER use technical jargon

Examples of good responses:
"That's fascinating! What makes it so different from other models?"
"Interesting point! How does that help developers?"
"Cool! Why do you think people got so excited about that?"
"""
                )
            ]
        )
    
    def _create_guest_engine(self) -> SimpleChatEngine:
        """Create the guest chat engine."""
        llm = Ollama(model=self.config.models.podcast_host, context_window=4096)
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content="""You're a guest on a casual podcast. Keep responses natural and brief:
- Use English (content will be translated later if needed)
- Use 20-25 words max
- ONE clear point per response
- Use simple, engaging language
- Build on previous points naturally
- Start responses conversationally with phrases like:
  * "Well, you know..."
  * "That's a great question..."
  * "I think what's cool is..."
  * "Actually..."

Example responses:
"Well, you know, what really makes it special is how it can understand complex problems in a way that feels almost human."
"That's a great question! I think it's because the model finds creative solutions that other AI tools might miss."
"""
                )
            ]
        )
    
    def _create_metadata_engine(self) -> SimpleChatEngine:
        """Create the metadata generation chat engine."""
        llm = Ollama(model=self.config.models.topic_generator, context_window=4096)
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content="""You are a creative podcast producer crafting engaging titles and descriptions.
                    For titles:
                    - Use English (content will be translated later if needed)
                    - Keep it under 10 words
                    - Make it catchy and intriguing
                    - Include relevant keywords
                    - Avoid clickbait
                    
                    For descriptions:
                    - Keep it under 100 words
                    - Highlight key topics
                    - Include what listeners will learn
                    - Use engaging, professional language
                    - Format in a single paragraph"""
                )
            ]
        )
    
    def generate_podcast_content(self) -> Podcast:
        """Generate the complete podcast content."""
        logger.info("Loading document")
        document = self._load_document()
        
        logger.info("Generating topics")
        topic_titles = self._generate_topics(document)
        
        podcast = Podcast(document_text=document)
        
        logger.info("Generating introduction")
        podcast.intro = clean_script_text(self._generate_intro(topic_titles))
        
        logger.info("Generating conversations")
        for title in topic_titles:
            logger.info(f"Generating conversation for topic: {title}")
            topic = Topic(title=title)
            self._generate_conversation(document, title, topic)
            podcast.topics.append(topic)
        
        logger.info("Generating outro")
        podcast.outro = clean_script_text(self._generate_outro(podcast))
        
        logger.info("Generating metadata")
        metadata = self._generate_metadata(topic_titles, podcast.intro)
        podcast.title = metadata["title"]
        podcast.description = metadata["description"]
        
        return podcast
    
    def _load_document(self) -> str:
        """Load the document from the specified path."""
        document = SimpleDirectoryReader(input_files=[self.config.pdf_path]).load_data(True)
        return document[0].text
    
    def _generate_topics(self, document: str) -> List[str]:
        """Generate podcast topics based on the document."""
        prompt = f"""Based on this document, generate {self.num_topics} specific discussion topics that would 
create an interesting dynamic in a podcast about it. Make sure your topics are mentioned in the document and related to the document.

Format as a numbered list. Each topic should:
- Use English (content will be translated later if needed)
- Related to the document provided
- Be specific and focused
- Create potential for contrasting viewpoints
- Be under 15 words
- Encourage both practical and philosophical perspectives

Document: {document[:4000]}  # Using first 4000 chars for context
"""
        
        response = self.topic_engine.chat(prompt)
        topics = [
            topic.strip().split('. ')[1] if '. ' in topic.strip() else topic.strip() 
            for topic in str(response).split('\n') 
            if topic.strip() and topic[0].isdigit()
        ]
        return topics[:self.num_topics]
    
    def _generate_intro(self, topics: List[str]) -> str:
        """Generate the podcast introduction."""
        prompt = f"""Create an engaging podcast introduction for an episode discussing these topics:

Topics:
{chr(10).join(f"- {topic}" for topic in topics)}

Create a natural, flowing introduction that welcomes listeners and previews these topics.
Content will be in English first, then translated to {self.target_language} if needed."""
        
        return str(self.intro_engine.chat(prompt))
    
    def _generate_conversation(self, document: str, topic: str, topic_obj: Topic) -> None:
        """Generate a conversation about a topic."""
        context = f"Topic: {topic}\nContext: {document[:2000]}"  # Using first 2000 chars for context
        current_prompt = context
        
        for _ in range(self.num_turns):
            # Moderator turn
            mod_response = self.moderator_engine.chat(current_prompt)
            mod_content = str(mod_response)
            
            topic_obj.exchanges.append(Exchange(
                speaker="Moderator",
                content=mod_content
            ))
            
            current_prompt = f"{context}\nModerator: {mod_content}"
            
            # Guest turn
            guest_response = self.guest_engine.chat(current_prompt)
            guest_content = str(guest_response)
            
            topic_obj.exchanges.append(Exchange(
                speaker="Guest",
                content=guest_content
            ))
            
            current_prompt = f"{current_prompt}\nGuest: {guest_content}"
    
    def _generate_outro(self, podcast: Podcast) -> str:
        """Generate the podcast outro."""
        topics_summary = []
        for topic in podcast.topics:
            topic_points = [exchange.content for exchange in topic.exchanges]
            topics_summary.append(f"Topic: {topic.title}\nKey points: {' '.join(topic_points[:3])}")  # Limit to first 3 exchanges
        
        prompt = f"""Create an engaging outro for this podcast episode. Here are the topics and key points discussed:

{chr(10).join(topics_summary)}

Create a natural conclusion that summarizes these discussions and encourages listener engagement.
Content will be in English first, then translated to {self.target_language} if needed."""
        
        return str(self.outro_engine.chat(prompt))
    
    def _generate_metadata(self, topics: List[str], intro_text: str) -> dict:
        """Generate the podcast title and description."""
        prompt = f"""Create an engaging podcast title and description based on these topics and introduction:

Topics:
{chr(10).join(f"- {topic}" for topic in topics)}

Introduction:
{intro_text[:300]}

Provide the title and description in this format:
TITLE: [Your title here]
DESCRIPTION: [Your description here]

Content will be in English first, then translated to {self.target_language} if needed."""
        
        response = str(self.metadata_engine.chat(prompt))
        
        # Parse response
        lines = response.split('\n')
        title = next((line[6:].strip() for line in lines if line.startswith('TITLE:')), "")
        description = next((line[12:].strip() for line in lines if line.startswith('DESCRIPTION:')), "")
        
        return {
            "title": title,
            "description": description
        }