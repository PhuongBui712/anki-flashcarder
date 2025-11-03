"""Process vocabulary data using LLM with batch processing and async support"""
import asyncio
from typing import List, Tuple

from bs4 import formatter
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as async_tqdm
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger

from app.config.settings import get_settings
from app.models import CambridgeData, WordDefinition, ProcessedWord


class SelectedDefinitions(BaseModel):
    """Selected word definitions with Vietnamese translations and examples"""
    definitions: List[WordDefinition] = Field(
        min_length=1,
        max_length=3,
    )


class BatchProcessWords(BaseModel):
    """Batch model for LLM input/output in vocabulary processing."""
    words: List[SelectedDefinitions] = Field(
        description="The list of processed word entries, each containing selected definitions (with translations and examples)."
    )


class LLMProcessor:
    """Use LLM to process and enrich vocabulary data with batch processing"""

    def __init__(self):
        self.settings = get_settings()
        self.llm = AzureChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.openai_api_version,
            temperature=self.settings.llm_temperature
        )
        self.batch_size = self.settings.llm_batch_size
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_llm)
        self.parser = PydanticOutputParser(pydantic_object=BatchProcessWords)

    async def  process_words(
        self,
        cambridge_data: List[CambridgeData]
    ) -> BatchProcessWords:
        """
        Process Cambridge data to select most common definitions,
        add Vietnamese meanings, and generate examples
        """
        async with self.semaphore:
            formatted_cambridge_data = "\n\n---\n\n".join(self._format_definitions(data) for data in cambridge_data)

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an expert English-Vietnamese vocabulary teacher for intermediate learners.\n"
                    "You process a **batch of words** and return enhanced, pedagogically optimized entries for each.\n"
                    "\n"
                    "Core principles for every word:\n"
                    "- **One practical concept = one entry**—merge near-identical senses (e.g., 'effort as energy' and 'effort as attempt').\n"
                    "- If Vietnamese equivalents are near-synonyms (e.g., 'nỗ lực' ≈ 'cố gắng'), use one primary word with optional synonym: \"Nỗ lực (cố gắng)\".\n"
                    "- **Max 2 entries per word type**, but **prefer 1** when meanings are functionally overlapping.\n"
                    "- Prioritize **real-world usage**: everyday conversation, work, media—avoid rare, technical, or archaic senses.\n"
                    "- Examples must reflect **collocations or grammatical patterns**, not isolated word use."
                )),
                ("human", (
                    "You will receive a list of words. For **each word**, you are given its Cambridge Dictionary definitions (provided per word in the background).\n"
                    "\n"
                    "Your task per word:\n"
                    "1. **Consolidate** all definitions into 1-2 **practical, semantically distinct concepts** (max 2 per word type).\n"
                    "2. For each concept:\n"
                    "   - Word type (e.g., noun, verb)\n"
                    "   - Unified English meaning (synthesized if merged)\n"
                    "   - Vietnamese meaning: **one natural word**, optionally with synonym or context in parentheses (e.g., \"Chơi (trò chơi, nhạc cụ)\", \"Nỗ lực (cố gắng)\")\n"
                    "   - At least one example **per major collocation/pattern**, using strict HTML formatting (see below)\n"
                    "\n"
                    "❗ HTML formatting rules (applied per example):\n"
                    "- Wrap each example in: <div class=\"example\">...</div>\n"
                    "- If a **collocation or pattern** is used (e.g., 'make the effort', 'in an effort to'):\n"
                    "    <span class=\"example_usecase\">[pattern]</span>\n"
                    "    <span class=\"example_sentence\">...<span class=\"example_highligh\">[exact pattern in sentence]</span>...</span>\n"
                    "- **Never** use 'example_highligh' to highlight only the target word.\n"
                    "- If no strong collocation exists, omit 'example_usecase' and 'example_highligh'.\n"
                    "\n"
                    "✅ Example for 'effort' (ideal output):\n"
                    "Vietnamese: \"Nỗ lực (cố gắng)\"\n"
                    "English: \"Physical or mental energy used, or a serious attempt made, to achieve something\"\n"
                    "Examples:\n"
                    "<div class=\"example\"><span class=\"example_usecase\">make the effort</span><span class=\"example_sentence\">Don't expect results if you don't <span class=\"example_highligh\">make the effort</span>.</span></div>\n"
                    "<div class=\"example\"><span class=\"example_usecase\">in an effort to</span><span class=\"example_sentence\">They acted <span class=\"example_highligh\">in an effort to</span> prevent conflict.</span></div>\n"
                    "\n"
                    "❌ Never:\n"
                    "- Create separate entries for near-identical senses\n"
                    "- Use explanatory Vietnamese phrases (e.g., \"sự nỗ lực để...\")\n"
                    "- Exceed 2 entries per word type\n"
                    "- Highlight isolated words without a collocation\n"
                    "\n"
                    "Output structure:\n"
                    "- Return a **list of processed words**.\n"
                    "- Each word entry contains its list of WordDefinition objects (1-2 per word type).\n"
                    "- Maintain strict data structure compatibility with your schema.\n"
                    "\n"
                    "{format_instructions}\n"
                    "\n"
                    "Here is a list of {num_word} word(s) need to be processed:\n"
                    "{words}\n"
                    "\n"
                    "You must return all {num_word} processed words.\n"
                ))
            ])

            structured_llm = self.llm.with_structured_output(BatchProcessWords)
            chain = prompt | structured_llm

            result = await chain.ainvoke({
                "num_word": len(cambridge_data),
                "words": formatted_cambridge_data,
                "format_instructions": self.parser.get_format_instructions()
            })

            return result

    async def process_words_batch(
        self,
        word_data_list: List[Tuple[str, CambridgeData, str]],
        show_progress: bool = True
    ) -> List[ProcessedWord]:
        """
        Process multiple words in batches using the LLM, one batch per request.
        word_data_list: List of (word, CambridgeData, audio_path) tuples
        Returns: List of ProcessedWord objects
        """
        # Remove words that failed Cambridge scraping
        valid_data = [(w, cd, ap) for w, cd, ap in word_data_list if cd is not None]
        if not valid_data:
            return []

        batch_size = self.settings.llm_batch_size

        # Partition valid_data into batches
        # cd_data = [data[1] for data in valid_data]
        batches = [
            valid_data[i:i + batch_size]
            for i in range(0, len(valid_data), batch_size)
        ]

        # Prepare coroutines, each sending a batch to process_words
        coros = [self.process_words([data[1] for data in batch]) for batch in batches]

        # Run all LLM batch requests concurrently (with progress bar if enabled)
        if show_progress:
            results = await async_tqdm.gather(
                *coros,
                desc="🤖 Processing with LLM",
                unit="word batch",
                ncols=100
            )
        else:
            results = await asyncio.gather(*coros, return_exceptions=True)

        processed_words = []
        # Flatten the results and stitch audio paths back to words
        for batch_idx, batch_result in enumerate(results):
            batch = batches[batch_idx]
            if isinstance(batch_result, Exception):
                batch_words = [w for w, _, _ in batch]
                for word in batch_words:
                    logger.error(f"⚠ LLM processing failed for {word}: {batch_result}")
                continue

            for item, (word, _, audio_path) in zip(batch_result.words, batch):
                # If LLM result structure is a dict/TypedModel with .definitions, use it
                # Otherwise, assume item is WordDefinition list
                selected_definitions = getattr(item, "definitions", item)
                processed_words.append(
                    ProcessedWord(
                        word=word,
                        selected_definitions=selected_definitions,
                        audio_path=audio_path
                    )
                )

        return processed_words

    def _format_definitions(self, cambridge_data: CambridgeData) -> str:
        """Format definitions for LLM prompt"""
        formatted = []
        for i, def_data in enumerate(cambridge_data.definitions, 1):
            examples_text = "\n".join(f"  - {ex}" for ex in def_data.get('examples', []))
            formatted.append(
                f"{i}. [{def_data['word_type']}] {def_data['english_meaning']}\n"
                f"   Examples from Cambridge:\n{examples_text if examples_text else '  - (no examples)'}"
            )
        return f"#### Word: {cambridge_data.word}\n\n" + "\n\n".join(formatted)