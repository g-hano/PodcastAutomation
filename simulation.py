"""Podcast simulation with LLM-powered host and guests."""

import time
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

import colorama
from colorama import Fore, Style
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine


from .utils.llm_factory import LLMFactory
from .utils.text_processing import clean_script_text
from .core.config import PodcastConfig
from .core.podcast import Podcast, Topic, Exchange

# Initialize colorama
colorama.init(autoreset=True)

class PodcastSimulation:
    """Simulates a podcast conversation using LLMs."""
    
    COLORS = {
        'Host': Fore.MAGENTA,
        'Moderator': Fore.BLUE,
        'Guest': Fore.GREEN,
        'System': Fore.YELLOW,
    }
    
    def __init__(self, config: PodcastConfig):
        """Initialize simulation with configuration."""
        self.config = config
        self.verbose = config.logging.verbose
        self.export_json = config.export_conversation_json
        self.num_turns = config.num_turns
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Topic generation, intro and outro
        self.topic_generator = self._create_topic_generator()
        self.intro_generator = self._create_intro_generator()
        self.outro_generator = self._create_outro_generator()
        self.metadata_generator = self._create_metadata_generator()
        
        # Conversation participants
        self.moderator = self._create_moderator()
        self.guest = self._create_guest()
        
        # Load document
        self.document = self._load_document()
        
        # Detailed conversation data for export
        self.conversation_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "document_path": config.pdf_path
            },
            "conversations": {}
        }
        
        # Track if subtitle generation is enabled
        self.generate_subtitles = config.audio.generate_subtitles
        self.subtitle_format = config.audio.subtitle_format
    
    
    def _load_document(self) -> str:
        """Load the document from the specified path."""
        try:
            document = SimpleDirectoryReader(input_files=[self.config.pdf_path]).load_data(True)
            return document[0].text
        except Exception as e:
            self._print_system(f"Error loading document: {str(e)}")
            raise
    
    def _create_topic_generator(self) -> SimpleChatEngine:
        """Create the topic generator."""
        llm = LLMFactory.create_llm(
            self.config.models.topic_generator,
            self.config.models,
            context_window=4096
        )
        
        return SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[
                ChatMessage(
                    role="system", 
                    content=f"""You are a creative podcast producer. Given a document about technology:
                    1. Use English (content will be translated later if needed)
                    2. Generate exactly {self.config.num_topics} specific, focused discussion topics
                    3. Topics should encourage debate and different perspectives
                    4. Each topic should be concrete and specific, not general
                    5. Focus on practical implications, technical innovations, and real-world impact
                    6. Each topic should be 10-15 words maximum"""
                )
            ]
        )
    
    def _create_intro_generator(self) -> SimpleChatEngine:
        """Create the introduction generator."""
        llm = LLMFactory.create_llm(
            self.config.models.intro_generator,
            self.config.models,
            context_window=4096
        )
        
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
    
    def _create_outro_generator(self) -> SimpleChatEngine:
        """Create the outro generator."""
        llm = LLMFactory.create_llm(
            self.config.models.outro_generator,
            self.config.models,
            context_window=4096
        )
        
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
    
    def _create_metadata_generator(self) -> SimpleChatEngine:
        """Create the metadata generator."""
        llm = LLMFactory.create_llm(
            self.config.models.topic_generator,
            self.config.models,
            context_window=4096
        )
        
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
    
    def _create_moderator(self) -> SimpleChatEngine:
        """Create the moderator."""
        llm = LLMFactory.create_llm(
            self.config.models.podcast_moderator,
            self.config.models,
            context_window=4096
        )
        
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
    
    def _create_guest(self) -> SimpleChatEngine:
        """Create the guest."""
        llm = LLMFactory.create_llm(
            self.config.models.podcast_guest,
            self.config.models,
            context_window=4096
        )
        
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
    
    def _print_message(self, speaker: str, content: str) -> None:
        """Print a message with color-coding."""
        color = self.COLORS.get(speaker, Style.RESET_ALL)
        if self.verbose:
            print(f"{color}{speaker}: {content}{Style.RESET_ALL}")
    
    def _print_system(self, message: str) -> None:
        """Print a system message."""
        if self.verbose:
            print(f"{self.COLORS['System']}{message}{Style.RESET_ALL}")
    
    def simulate_discussion(self, topic: str) -> List[Dict[str, Any]]:
        """Simulate a discussion about a topic."""
        self._print_system(f"=== Topic: {topic} ===\n")
        context = f"Topic: {topic}\nContext: {self.document[:4000]}"  # Limit context size
        
        current_prompt = context
        topic_exchanges = []
        
        for turn in range(self.num_turns):
            # Moderator turn
            mod_start_time = time.time()
            mod_response = self.moderator.chat(current_prompt)
            mod_content = str(mod_response)
            mod_end_time = time.time()
            
            exchange_data = {
                "speaker": "Moderator",
                "content": mod_content,
                "start_time": self._format_time(mod_start_time),
                "end_time": self._format_time(mod_end_time),
                "duration_seconds": round(mod_end_time - mod_start_time, 2),
                "turn": turn * 2  # 0, 2, 4...
            }
            
            self._print_message("Moderator", mod_content)
            topic_exchanges.append(exchange_data)
            
            current_prompt = f"{context}\nModerator: {mod_content}"
            time.sleep(random.uniform(0.5, 1.0))
            
            # Guest turn
            guest_start_time = time.time()
            guest_response = self.guest.chat(current_prompt)
            guest_content = str(guest_response)
            guest_end_time = time.time()
            
            exchange_data = {
                "speaker": "Guest",
                "content": guest_content,
                "start_time": self._format_time(guest_start_time),
                "end_time": self._format_time(guest_end_time),
                "duration_seconds": round(guest_end_time - guest_start_time, 2),
                "turn": turn * 2 + 1  # 1, 3, 5...
            }
            
            self._print_message("Guest", guest_content)
            topic_exchanges.append(exchange_data)
            
            current_prompt = f"{current_prompt}\nGuest: {guest_content}"
            time.sleep(random.uniform(0.5, 1.0))
        
        return topic_exchanges
    
    def _format_time(self, timestamp: float) -> str:
        """Format a timestamp into ISO format."""
        return datetime.fromtimestamp(timestamp).isoformat()
    
    def run_podcast_simulation(self) -> Podcast:
        """Run the complete podcast simulation."""
        self._print_system("Generating discussion topics...\n")
        topics = self._generate_topics()
        
        for i, topic in enumerate(topics, 1):
            self._print_system(f"{i}. {topic}")
        
        # Generate intro
        self._print_system("\n=== Episode Introduction ===\n")
        intro_text = clean_script_text(self._generate_intro(topics))
        self._print_message("Host", intro_text)
        
        # Create podcast object
        podcast = Podcast(document_text=self.document, intro=intro_text)
        
        # Main discussions
        for topic_title in topics:
            topic_obj = Topic(title=topic_title)
            
            # Simulate the discussion
            topic_exchanges = self.simulate_discussion(topic_title)
            
            # Save to podcast object
            for exchange_data in topic_exchanges:
                topic_obj.exchanges.append(Exchange(
                    speaker=exchange_data["speaker"],
                    content=exchange_data["content"],
                    timestamp=exchange_data["start_time"]
                ))
            
            # Save to conversation data for export
            self.conversation_data["conversations"][topic_title] = topic_exchanges
            
            # Add topic to podcast
            podcast.topics.append(topic_obj)
            
            time.sleep(1)
        
        # Generate outro
        self._print_system("\n=== Episode Conclusion ===\n")
        outro_text = clean_script_text(self._generate_outro(podcast))
        self._print_message("Host", outro_text)
        podcast.outro = outro_text
        
        # Generate metadata
        self._print_system("Generating podcast metadata...\n")
        metadata = self._generate_metadata(topics, intro_text)
        podcast.title = metadata["title"]
        podcast.description = metadata["description"]
        self._print_system(f"Title: {metadata['title']}")
        self._print_system(f"Description: {metadata['description']}\n")
        
        # Add metadata to conversation data
        self.conversation_data["metadata"]["title"] = metadata["title"]
        self.conversation_data["metadata"]["description"] = metadata["description"]
        self.conversation_data["intro"] = intro_text
        self.conversation_data["outro"] = outro_text
        
        # Save detailed conversation data if requested
        if self.export_json:
            self._export_conversation_data()
        
        return podcast
    
    def _generate_topics(self) -> List[str]:
        """Generate discussion topics."""
        prompt = f"""Based on this document, generate {self.config.num_topics} specific discussion topics that would 
create an interesting dynamic in a podcast about it. Make sure your topics are mentioned in the document and related to the document.

Format as a numbered list. Each topic should:
- Use English (content will be translated later if needed)
- Related to the document provided
- Be specific and focused
- Create potential for contrasting viewpoints
- Be under 15 words
- Encourage both practical and philosophical perspectives

Document: {self.document[:4000]}  # Using first 4000 chars for context
"""
        
        response = self.topic_generator.chat(prompt)
        topics = [
            topic.strip().split('. ')[1] if '. ' in topic.strip() else topic.strip() 
            for topic in str(response).split('\n') 
            if topic.strip() and topic[0].isdigit()
        ]
        return topics[:self.config.num_topics]
    
    def _generate_intro(self, topics: List[str]) -> str:
        """Generate the podcast introduction."""
        prompt = f"""Create an engaging podcast introduction for an episode discussing these topics:

Topics:
{chr(10).join(f"- {topic}" for topic in topics)}

Create a natural, flowing introduction that welcomes listeners and previews these topics."""
        
        return str(self.intro_generator.chat(prompt))
    
    def _generate_outro(self, podcast: Podcast) -> str:
        """Generate the podcast outro."""
        topics_summary = []
        for topic in podcast.topics:
            topic_points = [exchange.content for exchange in topic.exchanges]
            topics_summary.append(f"Topic: {topic.title}\nKey points: {' '.join(topic_points[:3])}")
        
        prompt = f"""Create an engaging outro for this podcast episode. Here are the topics and key points discussed:

{chr(10).join(topics_summary)}

Create a natural conclusion that summarizes these discussions and encourages listener engagement."""
        
        return str(self.outro_generator.chat(prompt))
    
    def _generate_metadata(self, topics: List[str], intro_text: str) -> Dict[str, str]:
        """Generate podcast title and description."""
        prompt = f"""Create an engaging podcast title and description based on these topics and introduction:

Topics:
{chr(10).join(f"- {topic}" for topic in topics)}

Introduction:
{intro_text[:300]}

Provide the title and description in this format:
TITLE: [Your title here]
DESCRIPTION: [Your description here]"""
        
        response = str(self.metadata_generator.chat(prompt))
        
        lines = response.split('\n')
        title = next((line[6:].strip() for line in lines if line.startswith('TITLE:')), "")
        description = next((line[12:].strip() for line in lines if line.startswith('DESCRIPTION:')), "")
        
        return {
            "title": title,
            "description": description
        }
    
    def _export_conversation_data(self) -> None:
        """Export the detailed conversation data to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create export directory if it doesn't exist
        export_path = self.output_dir / "exports"
        export_path.mkdir(exist_ok=True, parents=True)
        
        # Save JSON file
        json_path = export_path / f"podcast_conversation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_data, f, ensure_ascii=False, indent=2)
        
        self._print_system(f"Detailed conversation data exported to: {json_path}")
        
        # Generate subtitle files if requested
        if self.config.audio.generate_subtitles:
            self._generate_subtitle_files(export_path, timestamp)
    
    def _generate_subtitle_files(self, export_path: Path, timestamp: str) -> None:
        """Generate subtitle files from conversation data."""
        subtitle_format = self.subtitle_format
        
        if subtitle_format == "srt":
            self._generate_srt_subtitles(export_path, timestamp)
        elif subtitle_format == "vtt":
            self._generate_vtt_subtitles(export_path, timestamp)
    
    def _generate_srt_subtitles(self, export_path: Path, timestamp: str) -> None:
        """Generate SRT subtitle file."""
        try:
            srt_path = export_path / f"podcast_subtitles_{timestamp}.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                subtitle_index = 1
                running_time = 0  # Keep track of running time in seconds
                
                # Intro subtitle (assume 30 seconds for intro)
                if "intro" in self.conversation_data:
                    f.write(f"{subtitle_index}\n")
                    start_time = self._format_srt_time(0)
                    end_time = self._format_srt_time(30)
                    running_time = 30
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"Host: {self.conversation_data['intro']}\n\n")
                    subtitle_index += 1
                
                # Topic discussions
                for topic, exchanges in self.conversation_data["conversations"].items():
                    if not exchanges:
                        continue
                        
                    # Topic header (2 seconds)
                    f.write(f"{subtitle_index}\n")
                    start_time = self._format_srt_time(running_time)
                    running_time += 2
                    end_time = self._format_srt_time(running_time)
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"Topic: {topic}\n\n")
                    subtitle_index += 1
                    
                    # Exchanges
                    for exchange in exchanges:
                        f.write(f"{subtitle_index}\n")
                        # Use the duration in the exchange or a default of 10 seconds
                        duration = exchange.get("duration_seconds", 10)
                        start_time = self._format_srt_time(running_time)
                        running_time += duration
                        end_time = self._format_srt_time(running_time)
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{exchange['speaker']}: {exchange['content']}\n\n")
                        subtitle_index += 1
                        
                        # Add a small pause between exchanges (0.5 seconds)
                        running_time += 0.5
                
                # Outro subtitle (30 seconds)
                if "outro" in self.conversation_data:
                    f.write(f"{subtitle_index}\n")
                    start_time = self._format_srt_time(running_time)
                    running_time += 30
                    end_time = self._format_srt_time(running_time)
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"Host: {self.conversation_data['outro']}\n\n")
            
            self._print_system(f"SRT subtitles exported to: {srt_path}")
            
        except Exception as e:
            self._print_system(f"Error generating SRT subtitles: {str(e)}")
            logging.error(f"Error generating SRT subtitles: {str(e)}", exc_info=True)
    
    def _generate_vtt_subtitles(self, export_path: Path, timestamp: str) -> None:
        """Generate WebVTT subtitle file."""
        try:
            vtt_path = export_path / f"podcast_subtitles_{timestamp}.vtt"
            with open(vtt_path, 'w', encoding='utf-8') as f:
                # WebVTT header
                f.write("WEBVTT\n\n")
                running_time = 0  # Keep track of running time in seconds
                
                # Intro subtitle (assume 30 seconds for intro)
                if "intro" in self.conversation_data:
                    start_time = self._format_vtt_time(0)
                    end_time = self._format_vtt_time(30)
                    running_time = 30
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"Host: {self.conversation_data['intro']}\n\n")
                
                # Topic discussions
                for topic, exchanges in self.conversation_data["conversations"].items():
                    if not exchanges:
                        continue
                        
                    # Topic header (2 seconds)
                    start_time = self._format_vtt_time(running_time)
                    running_time += 2
                    end_time = self._format_vtt_time(running_time)
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"Topic: {topic}\n\n")
                    
                    # Exchanges
                    for exchange in exchanges:
                        # Use the duration in the exchange or a default of 10 seconds
                        duration = exchange.get("duration_seconds", 10)
                        start_time = self._format_vtt_time(running_time)
                        running_time += duration
                        end_time = self._format_vtt_time(running_time)
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{exchange['speaker']}: {exchange['content']}\n\n")
                        
                        # Add a small pause between exchanges (0.5 seconds)
                        running_time += 0.5
                
                # Outro subtitle (30 seconds)
                if "outro" in self.conversation_data:
                    start_time = self._format_vtt_time(running_time)
                    running_time += 30
                    end_time = self._format_vtt_time(running_time)
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"Host: {self.conversation_data['outro']}\n\n")
            
            self._print_system(f"WebVTT subtitles exported to: {vtt_path}")
            
        except Exception as e:
            self._print_system(f"Error generating WebVTT subtitles: {str(e)}")
            logging.error(f"Error generating WebVTT subtitles: {str(e)}", exc_info=True)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (00:00:00,000)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format seconds to WebVTT time format (00:00:00.000)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"