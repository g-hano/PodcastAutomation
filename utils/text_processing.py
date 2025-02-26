"""Text processing utilities for podcast generation."""


def clean_script_text(text: str) -> str:
    """Remove common prefixes and formatting from generated script text."""
    # List of patterns to remove
    unwanted_prefixes = [
        "Here is a possible intro:",
        "Here's a possible intro:",
        "Here is an intro:",
        "Here's an intro:",
        "Intro script:",
        "Here is the intro:",
        "Here's the intro:",
        "Here is outro script:",
        "Here's outro script:",
        "Here is the outro:",
        "Here's the outro:",
        "Outro script:",
        "Here is a possible outro:",
        "Here's a possible outro:"
    ]
    
    # Clean the text
    cleaned_text = text.strip()
    for prefix in unwanted_prefixes:
        if cleaned_text.lower().startswith(prefix.lower()):
            cleaned_text = cleaned_text[len(prefix):].strip()
            # Remove any leading colons or quotes
            cleaned_text = cleaned_text.lstrip('":')
            break
    
    return cleaned_text.strip()