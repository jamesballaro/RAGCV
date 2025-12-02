import regex as re
import tiktoken

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_encoding(encoding_name: Optional[str] = None):
    try:
        if encoding_name:
            return tiktoken.get_encoding(encoding_name)
        # Prefer model-based encoding if available; fall back to cl100k_base.
        try:
            return tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        # If tiktoken fails for any reason, raise an explicit error so caller
        raise RuntimeError("tiktoken initialisation failed") from e

def token_count(text: str, encoding=None) -> int:
    if not text:
        return 0
    enc = encoding or get_encoding()
    return len(enc.encode(text))

def sentence_tokenize(text: str) -> List[str]:
    cleaned = text.replace("\r", " ").strip()
    if not cleaned:
        return []
    # more robust splitter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def chunk_sentences_token_budget(sentences: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = token_count(sent)
        if current and current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current).strip())

            # compute overlap sentences
            overlap = []
            overlap_acc = 0
            for s in reversed(current):
                t = token_count(s)
                if overlap_acc + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_acc += t

            current = overlap.copy()
            current_tokens = sum(token_count(s) for s in current)

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if len(c) > 0]

def chunk_notes_sentences(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    sentences = sentence_tokenize(text)
    return chunk_sentences_token_budget(sentences, max_tokens, overlap_tokens)

def default_chunker(self, text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
    )
    return splitter.split_text(text)

def token_length(text: str) -> int:
    return token_count(text)

def split_bullets(text: str) -> List[str]:
    lines = text.splitlines()
    bullets = []
    current = []
    for ln in lines:
        if re.match(r'^(\s*[-•*]|\s*\d+\.)\s+', ln):
            if current:
                bullets.append(" ".join(current).strip())
            current = [ln.strip()]
        else:
            if current:
                current.append(ln.strip())
            else:
                current = [ln.strip()]
    if current:
        bullets.append(" ".join(current).strip())
    return [b for b in bullets if len(b) > 0]

def chunk_cv(self, text: str) -> List[str]:
    bullets = self._split_bullets(text)
    if not bullets:
        return self._default_chunker(text)

    max_tokens = 450
    overlap_tokens = 80

    chunks = []
    current = []
    current_tokens = 0

    for b in bullets:
        bt = token_count(b)
        if current and current_tokens + bt > max_tokens:
            chunks.append(" ".join(current).strip())

            # overlap last bullet
            overlap = []
            acc = 0
            for x in reversed(current):
                tx = token_count(x)
                if acc + tx > overlap_tokens:
                    break
                overlap.insert(0, x)
                acc += tx

            current = overlap.copy()
            current_tokens = sum(token_count(x) for x in current)

        current.append(b)
        current_tokens += bt

    if current:
        chunks.append(" ".join(current).strip())

    return chunks

def chunk_cover_letter(self, text: str) -> List[str]:
    sentences = self._sentence_tokenize(text)
    return self._chunk_sentences_token_budget(
        sentences,
        max_tokens=450,
        overlap_tokens=120,
    )

def chunk_notes(self, text: str) -> List[str]:
    return self._chunk_notes_sentences(
        text,
        max_tokens=400,
        overlap_tokens=120,
    )
