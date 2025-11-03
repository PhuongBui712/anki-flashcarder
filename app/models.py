"""Pydantic models for vocabulary processing"""
from typing import List, Optional
from pydantic import BaseModel, Field


class WordDefinition(BaseModel):
    """Definition for a single word type"""
    word_type: str = Field(description="Part of speech (noun, verb, adjective, etc.)")
    english_meaning: str = Field(description="English definition from Cambridge")
    vietnamese_meaning: str = Field(description="Vietnamese meaning for this english ")
    phonetic: Optional[str] = Field(default=None, description="Phonetic transcription (US)")
    examples: List[str] = Field(default_factory=list, description="Example sentences in HTML div format")


class VocabularyEntry(BaseModel):
    """Complete vocabulary entry for export"""
    word: str
    type: str  # Main word type
    cloze: str  # Cloze deletion format
    phonetic: str
    audio: str  # Path to audio file
    vietnamese_meaning: str
    english_meaning: str
    example: str  # HTML formatted examples


class CambridgeData(BaseModel):
    """Raw data scraped from Cambridge dictionary"""
    word: str
    definitions: List[dict]  # List of {word_type, meaning, phonetic, examples}
    audio_url: Optional[str] = None
    phonetic_us: Optional[str] = None


class ProcessedWord(BaseModel):
    """Processed word with selected definitions"""
    word: str
    selected_definitions: List[WordDefinition] = Field(
        min_length=1,
        max_length=5,
        description="3-5 most common word types and meanings"
    )
    audio_path: str
