"""Generate CSV output from processed vocabulary data"""
import csv
from typing import List
from pathlib import Path

from app.models import ProcessedWord, VocabularyEntry
from app.utils import create_cloze


class CSVGenerator:
    """Generate final CSV output with all vocabulary data"""

    def generate_entries(self, processed_words: List[ProcessedWord]) -> List[VocabularyEntry]:
        """Convert processed words to vocabulary entries for export"""
        entries = []

        for word_data in processed_words:
            # Combine all definitions into single entry
            all_types = []
            all_vietnamese = []
            all_english = []
            all_examples = []

            for definition in word_data.selected_definitions:
                all_types.append(definition.word_type)
                all_vietnamese.append(definition.vietnamese_meaning)
                all_english.append(definition.english_meaning)
                all_examples.extend(definition.examples)

            # Use the first definition's phonetic if available
            phonetic = word_data.selected_definitions[0].phonetic if word_data.selected_definitions else ""
            if not phonetic:
                # Try to get phonetic from any definition
                for def_item in word_data.selected_definitions:
                    if def_item.phonetic:
                        phonetic = def_item.phonetic
                        break

            entry = VocabularyEntry(
                word=word_data.word,
                type=", ".join(all_types),  # Join all word types
                cloze=create_cloze(word_data.word),
                phonetic=phonetic or "",
                audio=word_data.audio_path,
                vietnamese_meaning=" | ".join(all_vietnamese),  # Separate meanings with |
                english_meaning=" | ".join(all_english),  # Separate meanings with |
                example="\n".join(all_examples)  # Each example on new line
            )

            entries.append(entry)

        return entries

    def export_to_csv(self, entries: List[VocabularyEntry], output_path: str):
        """Export vocabulary entries to CSV file"""
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'word',
                'type',
                'cloze',
                'phonetic',
                'audio',
                'vietnamese_meaning',
                'english_meaning',
                'example'
            ])

            # Write data rows
            for entry in entries:
                writer.writerow([
                    entry.word,
                    entry.type,
                    entry.cloze,
                    entry.phonetic,
                    entry.audio,
                    entry.vietnamese_meaning,
                    entry.english_meaning,
                    entry.example
                ])

    def generate_and_export(
        self,
        processed_words: List[ProcessedWord],
        output_path: str
    ) -> str:
        """Generate entries and export to CSV in one step"""
        entries = self.generate_entries(processed_words)
        self.export_to_csv(entries, output_path)
        return output_path
