# Podcast Generator

An AI-driven podcast generation tool that creates professional audio podcasts from PDF documents in multiple languages.

## Features

- 🔍 Automatically extracts key topics from PDFs
- 💬 Generates engaging discussions between virtual hosts
- 🌐 Automatic translation to 9 different languages
- 🎙️ High-quality text-to-speech conversion using Kokoro
- 🎭 Support for multiple LLM providers (Ollama, OpenAI, Anthropic, Groq)
- 📊 Detailed conversation exports with timestamps
- 📝 Optional subtitle generation in SRT or VTT format
- 🎵 Background music integration
- 🔧 Customizable voices and audio settings

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/podcast-generator.git
cd podcast-generator

# Install dependencies
pip install -r requirements.txt

# Install Kokoro with language support as needed
pip install kokoro
pip install misaki[ja]  # For Japanese support
pip install misaki[zh]  # For Chinese support
```

## Configuration

Create a `config.yaml` file with the following structure:

```yaml
pdf_path: "path/to/your/document.pdf"
num_topics: 5
num_turns: 6
output_dir: "output"
export_conversation_json: true  # Export detailed conversation data with timestamps

models:
  topic_generator: "ollama/qwen2.5:14b"     # Format: provider/model_name
  podcast_moderator: "anthropic/claude-3-haiku-20240307"
  podcast_host: "ollama/llama3.1:8b" 
  intro_generator: "openai/gpt-3.5-turbo"
  outro_generator: "groq/llama3-8b-8192"
  translator: "ollama/qwen2.5:14b"
  providers:
    openai_api_key: "sk-..."       # Your OpenAI API key
    anthropic_api_key: "sk-ant-..." # Your Anthropic API key
    groq_api_key: "gsk_..."         # Your Groq API key
    ollama_base_url: "http://localhost:11434"  # Ollama server URL

audio:
  lang: "e"  # Language code determines the output language (see Language Codes section)
  host_voice: "ef_dora"       # Voice name for the host
  moderator_voice: "em_alex"  # Voice name for the moderator
  guest_voice: "em_santa"     # Voice name for the guest
  output_dir: "output"
  output_file: "podcast.wav"
  chunk_size: 200
  music_path: "music/background.mp3"
  vocal_volume: 0  # in dB
  bg_intro_volume: -12  # optional
  bg_content_volume: -20  # optional
  bg_outro_volume: -12
  generate_subtitles: true   # Generate subtitle files
  subtitle_format: "srt"     # "srt" or "vtt"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/podcast_generator.log"
  verbose: true  # Show colored conversation in console during generation
```

### LLM Provider Configuration

You can specify the LLM provider for each model by using the format `provider/model_name`:

- `ollama/model_name` - Use Ollama with local models (default if no provider specified)
- `openai/model_name` - Use OpenAI API (requires API key)
- `anthropic/model_name` - Use Anthropic API (requires API key)
- `groq/model_name` - Use Groq API (requires API key)

Example model specifications:
```yaml
models:
  topic_generator: "ollama/qwen2.5:14b"
  podcast_moderator: "anthropic/claude-3-haiku-20240307"
  podcast_host: "openai/gpt-3.5-turbo"
  intro_generator: "groq/mixtral-8x7b-32768"
```

### Language Codes

The `lang` parameter in the audio configuration determines both the target language for translation and the TTS voice language:

- `a`: American English (default)
- `b`: British English
- `j`: Japanese
- `h`: Hindi
- `p`: Portuguese
- `z`: Chinese (Mandarin)
- `i`: Italian
- `f`: French
- `e`: Spanish

### Export and Subtitle Options

```yaml
export_conversation_json: true  # Export detailed conversation data with timestamps

audio:
  generate_subtitles: true   # Generate subtitle files
  subtitle_format: "srt"     # "srt" or "vtt"
```

## Usage

### Basic Usage

```bash
python -m src.cli --config config.yaml
```

### Additional Options

```bash
# Specify PDF file
python -m src.cli --config config.yaml --pdf path/to/document.pdf

# Set output directory
python -m src.cli --config config.yaml --output-dir podcasts/my_podcast

# Set number of topics and turns
python -m src.cli --config config.yaml --topics 3 --turns 4

# Skip specific stages
python -m src.cli --config config.yaml --skip-translation --skip-audio

# Set logging options
python -m src.cli --config config.yaml --log-level DEBUG --log-file custom_log.log
```

## Export Formats

### Conversation JSON

When `export_conversation_json` is enabled, a detailed JSON file with all conversation data is generated:

```json
{
  "metadata": {
    "timestamp": "2023-03-15T14:22:33.123456",
    "document_path": "path/to/document.pdf",
    "title": "Podcast Title",
    "description": "Podcast description..."
  },
  "intro": "Welcome to our podcast...",
  "conversations": {
    "Topic 1": [
      {
        "speaker": "Moderator",
        "content": "What do you think about...",
        "start_time": "2023-03-15T14:22:40.123456",
        "end_time": "2023-03-15T14:22:45.123456",
        "duration_seconds": 5.0,
        "turn": 0
      },
      {
        "speaker": "Guest",
        "content": "Well, I think...",
        "start_time": "2023-03-15T14:22:46.123456",
        "end_time": "2023-03-15T14:22:52.123456",
        "duration_seconds": 6.0,
        "turn": 1
      }
    ]
  },
  "outro": "Thanks for listening..."
}
```

### Subtitle Files

When `generate_subtitles` is enabled, subtitle files are generated in either SRT or VTT format:

#### SRT Example
```
1
00:00:00,000 --> 00:00:30,000
Host: Welcome to our podcast...

2
00:00:30,000 --> 00:00:32,000
Topic: First Topic

3
00:00:32,000 --> 00:00:37,000
Moderator: What do you think about...
```

#### VTT Example
```
WEBVTT

00:00:00.000 --> 00:00:30.000
Host: Welcome to our podcast...

00:00:30.000 --> 00:00:32.000
Topic: First Topic

00:00:32.000 --> 00:00:37.000
Moderator: What do you think about...
```

## License

MIT License