"""
Anki Flashcard Generator - Main Entry Point

Generate English vocabulary flashcards with:
- Word, type, cloze, phonetic, audio, meanings, and examples
- Data from Cambridge Dictionary
- AI-enhanced content generation
"""
import sys
import traceback
import argparse
from pathlib import Path

from loguru import logger

from app.pipeline import VocabularyPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate English vocabulary flashcards using AI and Cambridge Dictionary"
    )

    # Create mutually exclusive group for input type
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--topic",
        type=str,
        help="Topic to generate vocabulary from (e.g., 'Education', 'Travel')"
    )
    input_group.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file containing words (first column)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV file path (optional, auto-generated if not provided)"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VocabularyPipeline()

    try:
        if args.topic:
            # Process from topic
            output_path = pipeline.process_from_topic(args.topic, args.output)
        else:
            # Process from CSV
            if not Path(args.csv).exists():
                print(f"Error: CSV file not found: {args.csv}")
                sys.exit(1)
            output_path = pipeline.process_from_csv(args.csv, args.output)

        print("\n" + "="*50)
        print("✓ SUCCESS!")
        print(f"Flashcards generated: {output_path}")
        print("="*50)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
