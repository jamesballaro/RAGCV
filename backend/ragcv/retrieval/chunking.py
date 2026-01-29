import re
import tiktoken
import spacy
from spacy.language import Language
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

_nlp_cache: Optional[Language] = None

def get_spacy_model() -> Language:
    """Lazy load spaCy model with sentencizer."""
    global _nlp_cache
    if _nlp_cache is None:
        try: 
            _nlp_cache = spacy.load(
                "en_core_web_sm",
                disable=["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]
            )
            if "sentencizer" not in _nlp_cache.pipe_names:
                _nlp_cache.add_pipe("sentencizer")
        except OSError as e:
            raise RuntimeError(
                "SpaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
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
    """Split text into sentences using spaCy's sentencizer."""
    cleaned = text.replace("\r", " ").strip()
    if not cleaned:
        return []
    
    nlp = get_spacy_model()
    doc = nlp(cleaned)
    
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            sentences.append(sent_text)
    
    return sentences

def split_bullets(text: str) -> List[str]:
    """
    Split text by bullet points or numbered lists.
    Handles nested bullets and continuations.
    """
    lines = text.splitlines()
    bullets = []
    current = []
    
    bullet_pattern = re.compile(r'^\s*[-•*]\s+|\s*\d+\.\s+')
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        if bullet_pattern.match(line):
            if current:
                bullets.append(" ".join(current))
            current = [line_stripped]
        else:
            if current:
                current.append(line_stripped)
            else:
                current = [line_stripped]
    
    if current:
        bullets.append(" ".join(current))
    
    return [b for b in bullets if b]

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

def chunk_notes(
    text: str, 
    ) -> List[str]:

    return sentence_tokenize(text)

@dataclass
class ChunkMetadata:
    chunk_id: str
    chunk_type: str
    section: str
    token_count: int
    source_document: Optional[str] = None
    # Flexible fields for specific types
    company: str = ""; role: str = ""; dates: str = ""
    start_date: Optional[str] = None; end_date: Optional[str] = None
    bullets: List[str] = field(default_factory=list)
    category: str = ""; skills_list: List[str] = field(default_factory=list)

class CVSemanticChunker:
    def __init__(self, max_tokens: int = 800):
        self.max_tokens = max_tokens
        self.enc = get_encoding()
        self.sect_re = re.compile(r'^(EXPERIENCE|EDUCATION|PROJECTS|SKILLS|SUMMARY|PROFILE)', re.I | re.M)
        self.date_re = re.compile(r'(\d{2}/\d{4}|\d{4})\s*[-–—]\s*(\d{2}/\d{4}|\d{4}|Present|Current)', re.I)

    def chunk_cv(self, text: str, source: str = None) -> List[Dict]:
        chunks = []
        # Split into sections
        matches = list(self.sect_re.finditer(text))
        if not matches: return self._create_chunks([text], "CONTENT", "fallback", source)

        for i, m in enumerate(matches):
            name, start = m.group(1).upper(), m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[start:end].strip()
            
            if name in ['EXPERIENCE', 'PROJECTS']:
                chunks.extend(self._process_jobs(content, name, source, len(chunks)))
            elif name == 'SKILLS':
                chunks.extend(self._process_skills(content, name, source, len(chunks)))
            else:
                chunks.extend(self._create_chunks([content], name, "section", source, len(chunks)))
        return chunks

    def _process_jobs(self, text: str, sect: str, src: str, start_idx: int) -> List[Dict]:
        job_chunks = []
        dates = list(self.date_re.finditer(text))
        if not dates: return self._create_chunks([text], sect, "job", src, start_idx)

        for i, m in enumerate(dates):
            job_start = max(0, m.start() - 150) # Catch header
            job_end = dates[i+1].start() if i+1 < len(dates) else len(text)
            job_text = text[job_start:job_end].strip()
            
            # Extract header info
            header = job_text.split('\n')[0]
            parts = [p.strip() for p in header.replace(m.group(0), "").split(',')]
            comp, role = (parts[0], parts[1]) if len(parts) >= 2 else ("", parts[0] if parts else "")
            
            job_chunks.extend(self._create_chunks([job_text], sect, "job", src, start_idx + len(job_chunks), 
                                                 company=comp, role=role, dates=m.group(0)))
        return job_chunks

    def _process_skills(self, text: str, sect: str, src: str, idx: int) -> List[Dict]:
        # Split by lines containing colons (categories)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return self._create_chunks(lines, sect, "skills", src, idx)

    def _create_chunks(self, items: List[str], sect: str, c_type: str, src: str, start_idx: int, **extra) -> List[Dict]:
        res = []
        curr_text, curr_tokens = [], 0
        
        for item in items:
            tokens = token_count(item, self.enc)
            if curr_text and (curr_tokens + tokens > self.max_tokens):
                res.append(self._finalize(curr_text, sect, c_type, src, f"{start_idx}_{len(res)}", curr_tokens, **extra))
                curr_text, curr_tokens = [], 0
            curr_text.append(item); curr_tokens += tokens
            
        if curr_text:
            res.append(self._finalize(curr_text, sect, c_type, src, f"{start_idx}_{len(res)}", curr_tokens, **extra))
        return res

    def _finalize(self, texts: List[str], sect: str, c_type: str, src: str, cid: str, tks: int, **extra) -> Dict:
        full_text = "\n".join(texts)
        meta = ChunkMetadata(chunk_id=cid, chunk_type=c_type, section=sect, token_count=tks, source_document=src, **extra)
        if c_type == "job": meta.bullets = split_bullets(full_text)
        return {"text": full_text, "metadata": asdict(meta)}

class CoverLetterChunker:
    """
    Chunk cover letters by paragraph with type classification.
    
    Improvements:
    - Better paragraph type detection
    - Token counting in metadata
    - Handles multi-paragraph intros/closings
    """
    
    def __init__(self):
        self.encoding = get_encoding()
    
    def chunk_cover_letter(
        self, 
        letter_text: str, 
    ) -> List[Dict[str, any]]:
        """
        Returns chunks with metadata.
        
        Returns:
            [
                {
                    'text': 'paragraph text...',
                    'metadata': CoverLetterChunkMetadata(...),
                },
                ...
            ]
        """
        # Split by double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in letter_text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        chunks = []
        
        for i, para in enumerate(paragraphs):            
            metadata = ChunkMetadata(
                chunk_id=f"cl_{i}",
                chunk_type="cover_letter_paragraph",
                section="COVER_LETTER",
                token_count=token_count(para, self.encoding),
            )
            
            chunks.append({
                'text': para,
                'metadata': metadata
            })
        
        return chunks
