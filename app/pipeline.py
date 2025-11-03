"""Main pipeline to orchestrate the vocabulary generation process with async support"""
import asyncio
import csv
from pathlib import Path
from typing import List, Optional
from loguru import logger

from app.config.settings import get_settings, ensure_directories
from app.topic_generator import TopicVocabularyGenerator
from app.cambridge_scraper import CambridgeScraper
from app.llm_processor import LLMProcessor
from app.csv_generator import CSVGenerator
from app.models import ProcessedWord


class VocabularyPipeline:
    """Main pipeline to generate vocabulary flashcards with async support"""

    def __init__(self):
        ensure_directories()
        self.settings = get_settings()
        self.topic_generator = TopicVocabularyGenerator()
        self.scraper = CambridgeScraper()
        self.llm_processor = LLMProcessor()
        self.csv_generator = CSVGenerator()

    def read_words_from_csv(self, csv_path: str) -> List[str]:
        """Read words from a CSV file (first column only)"""
        words = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if exists
            for row in reader:
                if row:  # Skip empty rows
                    words.append(row[0].strip().lower())
        return words

    async def process_from_topic_async(
        self,
        topic: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Process vocabulary generation from a topic (async).
        Steps: Topic -> Word List -> Process -> Output CSV
        """
        logger.info(f"📚 Generating vocabulary for topic: {topic}")

        # Step 1: Generate word list from topic
        temp_word_list = Path(self.settings.input_dir) / f"{topic.lower().replace(' ', '_')}_words.csv"
        self.topic_generator.generate_and_save(topic, str(temp_word_list))
        logger.success(f"✓ Generated word list: {temp_word_list}\n")

        # Step 2-4: Process the word list
        if not output_path:
            output_path = str(Path(self.settings.output_dir) / f"{topic.lower().replace(' ', '_')}_flashcards.csv")

        return await self.process_from_csv_async(str(temp_word_list), output_path)

    async def process_from_csv_async(
        self,
        csv_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Process vocabulary generation from a CSV file (async).
        Steps: CSV -> Scrape Cambridge -> LLM Process -> Output CSV
        """
        logger.info(f"📖 Reading words from: {csv_path}")

        # Step 1: Read words from CSV
        words = self.read_words_from_csv(csv_path)
        logger.info(f"Found {len(words)} words to process\n")

        # Step 2: Scrape Cambridge Dictionary concurrently with progress bar
        logger.info("=" * 60)
        scrape_results = await self.scraper.process_words_batch(words, show_progress=True)
        successful_scrapes = [(w, cd, ap) for w, cd, ap in scrape_results if cd is not None]
        logger.success(f"✓ Successfully scraped {len(successful_scrapes)}/{len(words)} words\n")

        if not successful_scrapes:
            logger.warning("⚠ No words were successfully scraped. Exiting.")
            return ""

        # Step 3: Process with LLM concurrently with progress bar
        logger.info("=" * 60)
        processed_words = await self.llm_processor.process_words_batch(
            successful_scrapes,
            show_progress=True
        )
        logger.success(f"✓ Successfully processed {len(processed_words)}/{len(successful_scrapes)} words with LLM\n")

        if not processed_words:
            logger.warning("⚠ No words were successfully processed by LLM. Exiting.")
            return ""

        # Step 4: Generate output CSV
        if not output_path:
            output_filename = Path(csv_path).stem + "_flashcards.csv"
            output_path = str(Path(self.settings.output_dir) / output_filename)

        self.csv_generator.generate_and_export(processed_words, output_path)
        logger.info("=" * 60)
        logger.success(f"💾 Output saved to: {output_path}")
        logger.info(f"📊 Final: {len(processed_words)}/{len(words)} words completed successfully")
        logger.info("=" * 60)

        return output_path

    def process_from_topic(
        self,
        topic: str,
        output_path: Optional[str] = None
    ) -> str:
        """Synchronous wrapper for process_from_topic_async"""
        return asyncio.run(self.process_from_topic_async(topic, output_path))

    def process_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """Synchronous wrapper for process_from_csv_async"""
        return asyncio.run(self.process_from_csv_async(csv_path, output_path))