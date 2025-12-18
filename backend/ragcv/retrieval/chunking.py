import tiktoken
import spacy
from spacy.language import Language
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Global spaCy model cache
_nlp_cache: Optional[Language] = None


def get_spacy_model() -> Language:
    """
    Lazy load spaCy model with sentencizer.
    Assumes en_core_web_sm is installed.
    """
    global _nlp_cache
    if _nlp_cache is None:
        # Load small model with only sentencizer (fast, accurate)
        _nlp_cache = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]
        )
        # Explicitly add sentencizer if not present
        if "sentencizer" not in _nlp_cache.pipe_names:
            _nlp_cache.add_pipe("sentencizer")
    return _nlp_cache


def get_encoding(encoding_name: Optional[str] = None):
    """Get tiktoken encoding for token counting."""
    try:
        if encoding_name:
            return tiktoken.get_encoding(encoding_name)
        try:
            return tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        raise RuntimeError("tiktoken initialization failed") from e


def token_count(text: str, encoding=None) -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    enc = encoding or get_encoding()
    return len(enc.encode(text))


def sentence_tokenize(text: str) -> List[str]:
    """
    Split text into sentences using spaCy's sentencizer.
    Much more accurate than regex for handling:
    - Abbreviations (Dr., Inc., etc.)
    - Decimal numbers (3.14)
    - Quotes and nested punctuation
    - Multiple sentence enders
    """
    cleaned = text.replace("\r", " ").strip()
    if not cleaned:
        return []
    
    nlp = get_spacy_model()
    
    # Process text in batches for efficiency if needed
    # For typical CV/cover letter text, single doc is fine
    doc = nlp(cleaned)
    
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:  # Only keep non-empty sentences
            sentences.append(sent_text)
    
    return sentences


def chunk_sentences_token_budget(
    sentences: List[str], 
    max_tokens: int, 
    overlap_tokens: int,
    encoding=None
) -> List[str]:
    """
    Chunk sentences respecting token budget with overlap.
    """
    if not sentences:
        return []
    
    enc = encoding or get_encoding()
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = token_count(sent, enc)
        
        # If adding this sentence exceeds budget, finalize current chunk
        if current and current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current))
            
            # Calculate overlap: work backwards through current chunk
            overlap = []
            overlap_acc = 0
            for s in reversed(current):
                s_tokens = token_count(s, enc)
                if overlap_acc + s_tokens > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_acc += s_tokens
            
            # Start new chunk with overlap
            current = overlap.copy()
            current_tokens = sum(token_count(s, enc) for s in current)
        
        current.append(sent)
        current_tokens += sent_tokens

    # Add final chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def split_bullets(text: str) -> List[str]:
    """
    Split text by bullet points or numbered lists.
    More robust than pure regex - handles nested bullets.
    """
    lines = text.splitlines()
    bullets = []
    current = []
    
    # Regex pattern for bullets: -, •, *, or numbered (1., 2., etc.)
    import re
    bullet_pattern = re.compile(r'^\s*[-•*]\s+|\s*\d+\.\s+')
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Check if line starts with bullet/number
        if bullet_pattern.match(line):
            # Save previous bullet if exists
            if current:
                bullets.append(" ".join(current))
            current = [line_stripped]
        else:
            # Continuation of current bullet
            if current:
                current.append(line_stripped)
            else:
                # Standalone line (header, etc.)
                current = [line_stripped]
    
    # Add final bullet
    if current:
        bullets.append(" ".join(current))
    
    return [b for b in bullets if b]


def chunk_cv(
    text: str, 
    max_tokens: int = 450, 
    overlap_tokens: int = 80
) -> List[str]:
    """
    Chunk CV text by bullet points with token budget.
    
    CVs are structured with bullet points, so we:
    1. Split by bullets first
    2. Group bullets into chunks respecting token budget
    3. Add overlap for context
    """
    bullets = split_bullets(text)
    
    if not bullets:
        # Fallback to default chunker if no bullets detected
        return default_chunker(text, chunk_size=max_tokens * 4, chunk_overlap=overlap_tokens * 4)

    enc = get_encoding()
    chunks = []
    current = []
    current_tokens = 0

    for bullet in bullets:
        bullet_tokens = token_count(bullet, enc)
        
        # If single bullet exceeds max, split it with sentence chunker
        if bullet_tokens > max_tokens:
            # Finalize current chunk first
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            
            # Split large bullet by sentences
            sentences = sentence_tokenize(bullet)
            sub_chunks = chunk_sentences_token_budget(
                sentences, 
                max_tokens, 
                overlap_tokens, 
                enc
            )
            chunks.extend(sub_chunks)
            continue
        
        # Check if adding bullet exceeds budget
        if current and current_tokens + bullet_tokens > max_tokens:
            chunks.append(" ".join(current))
            
            # Calculate overlap
            overlap = []
            overlap_acc = 0
            for b in reversed(current):
                b_tokens = token_count(b, enc)
                if overlap_acc + b_tokens > overlap_tokens:
                    break
                overlap.insert(0, b)
                overlap_acc += b_tokens
            
            current = overlap.copy()
            current_tokens = sum(token_count(b, enc) for b in current)
        
        current.append(bullet)
        current_tokens += bullet_tokens

    # Add final chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_cover_letter(
    text: str, 
    max_tokens: int = 450, 
    overlap_tokens: int = 120
) -> List[str]:
    """
    Chunk cover letter using sentence-aware splitting.
    """
    sentences = sentence_tokenize(text)
    return chunk_sentences_token_budget(
        sentences,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def chunk_notes(
    text: str, 
    max_tokens: int = 400, 
    overlap_tokens: int = 120
) -> List[str]:
    """
    Chunk notes using sentence-aware splitting.
    
    Notes can be mixed format, so use sentence splitting
    with moderate overlap.
    """
    sentences = sentence_tokenize(text)
    return chunk_sentences_token_budget(
        sentences,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def default_chunker(
    text: str, 
    chunk_size: int = 1024, 
    chunk_overlap: int = 256
) -> List[str]:
    """
    Fallback chunker using LangChain's RecursiveCharacterTextSplitter.
    Used when text structure is unknown or sentence splitting fails.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def token_length(text: str) -> int:
    """Convenience function for token counting."""
    return token_count(text)


def chunk_large_document(
    text: str,
    max_tokens: int = 450,
    overlap_tokens: int = 120,
    batch_size: int = 10000
) -> List[str]:
    """
    Chunk very large documents by processing in batches.
    Useful for 100+ page documents to avoid memory issues.
    """
    if len(text) <= batch_size:
        return chunk_notes(text, max_tokens, overlap_tokens)
    
    # Split into batches
    all_chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + batch_size, len(text))
        batch = text[start:end]
        
        # Process batch
        batch_chunks = chunk_notes(batch, max_tokens, overlap_tokens)
        all_chunks.extend(batch_chunks)
        
        # Move to next batch with overlap
        start = end - (overlap_tokens * 10)  # Rough estimate: 10 chars per token
    
    return all_chunks