"""Utility functions"""
import re


def create_cloze(word: str) -> str:
    """
    Create cloze deletion format for a word.
    Preserves spacing and special characters like hyphens.

    Example: 'ice-cream' -> '___-_____'
    """
    cloze = ""
    for char in word:
        if char.isalpha():
            cloze += "_"
        else:
            cloze += char
    return cloze


def clean_word(word: str) -> str:
    """Clean and normalize word"""
    return word.strip().lower()
