from transformers import pipeline
import logging

logger = logging.getLogger("JARVIS.Intent")

class IntentClassifier:
    def __init__(self):
        """
        We use a 'Zero-Shot Classification' model here.
        Instead of manually training a model on thousands of sentences, Zero-Shot allows us 
        to classify ANY spoken text into our custom categories dynamically using a pre-built language model!
        """
        logger.info("Loading Zero-Shot Intent Classifier...")
        # A lightweight, fast model for CPU inference. 
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        
        # These are the exact intents (commands) JARVIS is allowed to understand.
        self.candidate_labels = [
            "open application", 
            "get weather", 
            "web search", 
            "play media", 
            "system control",
            "casual conversation"
        ]
        logger.info("Intent Classifier ready.")

    def get_intent(self, user_text: str) -> str:
        """
        Takes raw spoken text (e.g., "What's the weather in London") and returns the intent ("get weather").
        """
        result = self.classifier(user_text, self.candidate_labels)
        
        # The AI returns a list ordered by highest confidence match first
        top_intent = result['labels'][0]
        confidence = result['scores'][0]
        
        logger.info(f"Intent classified as: {top_intent} (Confidence: {confidence:.2f})")
        return top_intent
