import spacy
import logging

logger = logging.getLogger("JARVIS.Entities")

class EntityExtractor:
    def __init__(self):
        """
        Loads the spaCy English NLP pipeline.
        This is used for Named Entity Recognition (NER), which automatically pulls out
        proper nouns like Cities, People, Dates, and Times from sentences.
        """
        logger.info("Loading spaCy NLP model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy Model loaded.")
        except OSError:
            logger.warning("Spacy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract(self, text: str) -> dict:
        """
        Extracts entities from the text.
        Example Input: "What is the weather in New York tomorrow?"
        Example Output: {"GPE": ["New York"], "DATE": ["tomorrow"]}
        """
        if not self.nlp:
            return {}

        doc = self.nlp(text)
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        logger.debug(f"Extracted entities: {entities}")
        return entities
