"""Generate vocabulary words from a given topic using LLM"""
import csv
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.config.settings import get_settings


class VocabularyList(BaseModel):
    """List of vocabulary words for a topic"""
    words: List[str] = Field(
        description="List of 20-30 common English vocabulary words related to the topic",
        min_length=20,
        max_length=30
    )


class TopicVocabularyGenerator:
    """Generate vocabulary words from topics using LLM"""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.azure_openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=VocabularyList)

    def generate_words_from_topic(self, topic: str) -> List[str]:
        """Generate vocabulary words for a given topic"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful English vocabulary teacher."),
            ("human", """Generate a list of 20-30 common and practical English vocabulary words
related to the topic: {topic}

The words should be:
- Commonly used in everyday English
- Relevant to the topic
- A mix of different word types (nouns, verbs, adjectives, etc.)
- Appropriate for intermediate English learners

{format_instructions}""")
        ])

        chain = prompt | self.llm | self.parser

        result = chain.invoke({
            "topic": topic,
            "format_instructions": self.parser.get_format_instructions()
        })

        return result.words

    def save_to_csv(self, words: List[str], output_path: str):
        """Save words to CSV file with single column"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word'])  # Header
            for word in words:
                writer.writerow([word.lower().strip()])

    def generate_and_save(self, topic: str, output_path: str) -> str:
        """Generate words from topic and save to CSV"""
        words = self.generate_words_from_topic(topic)
        self.save_to_csv(words, output_path)
        return output_path
