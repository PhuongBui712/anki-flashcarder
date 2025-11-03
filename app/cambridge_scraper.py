"""Scrape vocabulary data from Cambridge Dictionary with async support"""
import re
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import httpx
from bs4 import BeautifulSoup, NavigableString
from tqdm.asyncio import tqdm as async_tqdm
from loguru import logger

from app.config.settings import get_settings
from app.models import CambridgeData


class CambridgeScraper:
    """Scrape word definitions and audio from Cambridge Dictionary with async + semaphore"""

    BASE_URL = "https://dictionary.cambridge.org/dictionary/english"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self):
        self.settings = get_settings()
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_scrapes)

    async def scrape_word(self, word: str) -> Optional[CambridgeData]:
        """Scrape all data for a word from Cambridge Dictionary"""
        word = word.strip().lower()
        url = f"{self.BASE_URL}/{re.sub(r"[ \t]+", "-", word)}"

        async with self.semaphore:
            # Rate limiting
            await asyncio.sleep(self.settings.scraper_rate_limit)

            try:
                async with httpx.AsyncClient(timeout=30.0, headers=self.HEADERS) as client:
                    response = await client.get(url)
                    response.raise_for_status()
            except httpx.RequestError as e:
                logger.error(f"⚠ Error fetching {word}: {e}")
                return None
            except httpx.HTTPStatusError as e:
                logger.error(f"⚠ HTTP error fetching {word}: {e}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract phonetic (US)
            phonetic_us = self._extract_phonetic_us(soup)

            # Extract audio URL (US)
            audio_url = self._extract_audio_url_us(soup)

            # Extract definitions with examples
            definitions = self._extract_definitions(soup)

            if not definitions:
                logger.warning(f"⚠ No definitions found for: {word}")
                return None

            return CambridgeData(
                word=word,
                definitions=definitions,
                audio_url=audio_url,
                phonetic_us=phonetic_us
            )

    def _extract_phonetic_us(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract US phonetic transcription"""
        # Try to find US pronunciation
        us_pron = soup.find('span', class_='us')
        if us_pron:
            ipa = us_pron.find('span', class_='ipa')
            if ipa:
                return ipa.get_text(strip=True)

        # Fallback to any phonetic
        ipa = soup.find('span', class_='ipa')
        if ipa:
            return ipa.get_text(strip=True)

        return None

    def _extract_audio_url_us(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract US audio URL"""
        # Look for US audio
        us_audio = soup.find('span', class_='us')
        if us_audio:
            audio_tag = us_audio.find('source', type='audio/mpeg')
            if audio_tag and audio_tag.get('src'):
                url = audio_tag['src']
                if url.startswith('//'):
                    return 'https:' + url
                elif url.startswith('/'):
                    return 'https://dictionary.cambridge.org' + url
                return url

        # Fallback to any audio
        audio_tag = soup.find('source', type='audio/mpeg')
        if audio_tag and audio_tag.get('src'):
            url = audio_tag['src']
            if url.startswith('//'):
                return 'https:' + url
            elif url.startswith('/'):
                return 'https://dictionary.cambridge.org' + url
            return url

        return None

    def _get_examples(self, soup: BeautifulSoup, max_example: int = 3) -> Optional[str]:
        """Extract examples using the provided method"""
        examples_box = soup.find("div", "def-body ddef_b")

        if examples_box:
            example_tags = examples_box.find_all("div", "examp dexamp")
        else:
            example_tags = []

        if not examples_box or not example_tags:
            # Try to find examples in the expanded section
            expanded_section = soup.find("section", {"expanded": ""})
            if expanded_section:
                example_tags = expanded_section.find_all("li", {"class": "eg dexamp hax"})
            else:
                return None

        examples = []
        for i, exp in enumerate(example_tags):
            if i >= max_example:
                break
            exp_usecase = ""

            # handle usecase
            use_case = exp.find("span", "gram dgram") or exp.find("a", "lu dlu")
            if use_case:
                exp_usecase += f'<span class="example_usecase">{use_case.text}</span>'

            # handle example sentence
            exp_sentence = "  "
            if examples_box and exp.find("span", "eg deg"):
                exp_sentence_tag = exp.find("span", "eg deg")
                for child in exp_sentence_tag.children:
                    if isinstance(child, NavigableString):
                        exp_sentence += child.text
                    else:
                        if (
                            child.name == "span"
                            and child.attrs.get("class") == "b db".split()
                        ):
                            exp_sentence += (
                                f'<span class="example_highligh">{child.text}</span>'
                            )
                        else:
                            exp_sentence += child.text
            else:
                # Extract example sentence from expanded section
                exp_sentence += exp.get_text()

            exp_sentence = f'<span class="example_sentence">{exp_sentence}</span>'
            exp_tag = f'<div class="example">{exp_usecase}{exp_sentence}</div>'
            examples.append(exp_tag)

        return "\n".join(examples) if examples else None

    def _extract_definitions(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all definitions with word types, meanings, and examples"""
        definitions = []

        # Find all definition blocks
        entries = soup.find_all('div', class_='entry-body__el')

        for entry in entries:
            # Get word type (pos)
            pos_header = entry.find('div', class_='pos-header')
            if not pos_header:
                continue

            pos = pos_header.find('span', class_='pos')
            word_type = pos.get_text(strip=True) if pos else "unknown"

            # Find all sense blocks (different meanings)
            sense_blocks = entry.find_all('div', class_=re.compile(r'def-block'))

            for sense in sense_blocks:
                # Get definition
                def_tag = sense.find('div', class_='def')
                if not def_tag:
                    continue

                english_meaning = def_tag.get_text().strip()
                if english_meaning and not english_meaning[-1].isalpha():
                    english_meaning = english_meaning[:-1]

                # Get examples using the new method
                examples_str = self._get_examples(sense, max_example=3)
                examples = examples_str.split('\n') if examples_str else []

                definitions.append({
                    'word_type': word_type,
                    'english_meaning': english_meaning,
                    'examples': examples
                })

        return definitions

    async def download_audio(self, audio_url: str, word: str) -> str:
        """Download audio file and return local path"""
        if not audio_url:
            return ""

        # Create filename
        filename = f"{word}.mp3"
        filepath = Path(self.settings.audio_download_dir) / filename

        # Create directory if not exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=30.0, headers=self.HEADERS) as client:
                response = await client.get(audio_url)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                return str(filepath)
        except httpx.RequestError as e:
            logger.error(f"⚠ Error downloading audio for {word}: {e}")
            return ""

    async def process_word(self, word: str) -> Optional[Tuple[str, CambridgeData, str]]:
        """
        Scrape word data and download audio.
        Returns tuple of (word, CambridgeData, audio_path) or None
        """
        cambridge_data = await self.scrape_word(word)
        if not cambridge_data:
            return None

        audio_path = ""
        if cambridge_data.audio_url:
            audio_path = await self.download_audio(cambridge_data.audio_url, word)

        return (word, cambridge_data, audio_path)

    async def process_words_batch(
        self,
        words: List[str],
        show_progress: bool = True
    ) -> List[Tuple[str, Optional[CambridgeData], str]]:
        """
        Process multiple words concurrently with progress bar.
        Returns list of (word, CambridgeData, audio_path) tuples.
        Words that failed return (word, None, "").
        """
        tasks = [self.process_word(word) for word in words]

        if show_progress:
            results = await async_tqdm.gather(
                *tasks,
                desc="🔍 Scraping Cambridge",
                unit="word",
                ncols=100
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"⚠ Exception processing {words[i]}: {result}")
                processed_results.append((words[i], None, ""))
            elif result is None:
                processed_results.append((words[i], None, ""))
            else:
                processed_results.append(result)

        return processed_results