"""
Speech → Text → Summary → Search pipeline.

Two components:

1. ArabicSummarizer  — generates a concise summary from a (long) transcript.
   Default: csebuetnlp/mT5_multilingual_XLSum (43 languages including Arabic,
   trained on BBC News summarization). Beats AraBART on most Arabic eval sets
   and handles MSA + dialectal mix.

2. TranscriptSearch  — builds a semantic search index over a corpus of transcripts,
   so you can query "اجتماع الميزانية" and retrieve the matching audio clips.
   Uses sentence-transformers' multilingual MiniLM (works on Arabic).

Pipeline:
    audio → Whisper.transcribe_array(...) → ArabicSummarizer.summarize(text)
                                          ↘ TranscriptSearch.add(audio_id, text)
                                            TranscriptSearch.query("...") → top-k
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── 1. Summarization ──────────────────────────────────────────────────────────

class ArabicSummarizer:
    """mT5-based Arabic text summarization."""

    def __init__(
        self,
        model_name: str = "csebuetnlp/mT5_multilingual_XLSum",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading summarizer ({model_name}) on {self.device}...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Summarizer loaded.")

    @torch.no_grad()
    def summarize(
        self,
        text: str,
        max_length: int = 80,
        min_length: int = 20,
        num_beams: int = 4,
    ) -> str:
        """Return a short summary of the input text."""
        if not text or len(text.strip()) < 30:
            # Too short to summarize meaningfully
            return text.strip()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── 2. Semantic search over transcripts ───────────────────────────────────────

class TranscriptSearch:
    """In-memory semantic search index built from transcribed audio."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model

        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading embedding model ({embedding_model}) on {self.device}...")
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(embedding_model, device=self.device)

        # Storage
        self.entries: List[Dict] = []     # [{id, text, metadata}, ...]
        self.embeddings: Optional[np.ndarray] = None  # (N, dim)
        print("Search index ready.")

    def add(self, entry_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Add one transcript to the index. Re-encodes incrementally."""
        emb = self.encoder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        self.entries.append({
            "id": entry_id,
            "text": text,
            "metadata": metadata or {},
        })
        self.embeddings = emb if self.embeddings is None else np.vstack([self.embeddings, emb])

    def add_batch(self, items: List[Dict]) -> None:
        """Add many at once. Each item: {id, text, metadata?}"""
        texts = [it["text"] for it in items]
        embs = self.encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        self.entries.extend(items)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])

    def query(self, q: str, top_k: int = 5) -> List[Dict]:
        """Return top-k most-similar entries. Each entry includes its score."""
        if self.embeddings is None or not self.entries:
            return []
        q_emb = self.encoder.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self.embeddings @ q_emb  # cosine since both normalized
        idx = np.argsort(-scores)[:top_k]
        return [
            {**self.entries[i], "score": float(scores[i])}
            for i in idx
        ]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embedding_model": self.embedding_model_name,
            "entries": self.entries,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.entries = data["entries"]
        embs = data.get("embeddings", [])
        self.embeddings = np.array(embs) if embs else None

    def size(self) -> int:
        return len(self.entries)


# ── 3. Convenience pipeline ───────────────────────────────────────────────────

def build_search_corpus_from_audio_dir(
    audio_dir: str,
    asr_transcriber,        # any object with .transcribe_array(wav)
    summarizer: Optional[ArabicSummarizer] = None,
    search_index: Optional[TranscriptSearch] = None,
    max_files: int = 50,
) -> Tuple[TranscriptSearch, List[Dict]]:
    """
    Transcribe every audio file in `audio_dir` and (optionally) summarize + index it.
    Returns the search index plus a list of {file, transcript, summary} per item.
    """
    import torchaudio
    import torchaudio.functional as AF
    from pathlib import Path as _Path

    if search_index is None:
        search_index = TranscriptSearch()
    if summarizer is None:
        summarizer = ArabicSummarizer()

    audio_paths = sorted([
        p for p in _Path(audio_dir).rglob("*")
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
    ])[:max_files]

    records = []
    items_to_index = []
    for p in audio_paths:
        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(0).numpy().astype(np.float32)
        if sr != 16000:
            wav = AF.resample(torch.tensor(wav).unsqueeze(0), sr, 16000).squeeze().numpy()

        try:
            transcript = asr_transcriber.transcribe_array(wav)
        except Exception as e:
            transcript = f"[transcription failed: {e}]"
        summary = ""
        if summarizer and len(transcript.strip()) >= 30:
            try:
                summary = summarizer.summarize(transcript)
            except Exception:
                summary = transcript[:120]

        records.append({"file": str(p), "transcript": transcript, "summary": summary})
        items_to_index.append({
            "id": str(p),
            "text": transcript,
            "metadata": {"file": str(p), "summary": summary},
        })

    search_index.add_batch(items_to_index)
    return search_index, records
