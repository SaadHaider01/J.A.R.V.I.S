import threading
import time
import queue
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
import os

logger = logging.getLogger("ZYTRIX.AmbientAwareness")

@dataclass
class AmbientEvent:
    event_type: str       # "PHONE_RING", "KNOCK", "TIMER_ALARM"
    confidence: float
    timestamp: float
    duration_s: float

class AmbientAwarenessService:
    def __init__(self, audio_bus, clock=time):
        self.audio_bus = audio_bus
        self.clock = clock
        self.is_running = False
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        
        # Configuration
        self.window_seconds = 1.0
        self.hop_seconds = 3.0
        self.confidence_threshold = 0.50 # slightly relaxed for quantization
        self.event_cooldown_seconds = 10.0
        
        self.sample_rate = 16000
        
        # Debounce tracking
        self.last_event_times: Dict[str, float] = {}
        
        # Classification Mapping (whitelist)
        self.class_mapping = {
            "Telephone": "PHONE_RING",
            "Telephone bell ringing": "PHONE_RING",
            "Ringtone": "PHONE_RING",
            "Knock": "KNOCK",
            "Alarm clock": "TIMER_ALARM"
        }
        
        self.classifier = None
        self.classifier_ready = False
        
    def _init_classifier(self):
        try:
            logger.info("Initializing AST Ambient Classifier... (this may take a moment to load weights)")
            from transformers import pipeline
            import torch
            import warnings
            
            # Disable symlinks warning for Windows
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            
            self.classifier = pipeline(
                "audio-classification", 
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=-1 # CPU
            )
            
            # Apply Dynamic INT8 Quantization
            logger.info("Applying dynamic INT8 quantization for performance...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.classifier.model = torch.quantization.quantize_dynamic(
                    self.classifier.model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                
            self.classifier_ready = True
            logger.info("Ambient Classifier initialized successfully (INT8).")
        except Exception as e:
            logger.error(f"Failed to initialize Ambient Classifier: {e}")
            self.classifier_ready = False
            
    def start(self):
        self.shutdown_event.clear()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._run_loop, name="Ambient-Worker", daemon=True)
        self.worker_thread.start()
        
    def stop(self):
        self.is_running = False
        self.shutdown_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
    def _process_audio_window(self, audio_data: np.ndarray):
        if not self.classifier_ready or self.classifier is None:
            return
            
        import torch
        # Temporarily limit PyTorch threads to prevent global starvation of Whisper
        old_threads = torch.get_num_threads()
        
        try:
            torch.set_num_threads(2)
            # The pipeline expects sampling_rate, and either a numpy array or bytes. 
            predictions = self.classifier({"raw": audio_data, "sampling_rate": self.sample_rate})
        except Exception as e:
            logger.error(f"Ambient inference failed: {e}")
            return
        finally:
            torch.set_num_threads(old_threads)
            
        try:
            # predictions is a list of dicts: [{'score': 0.9, 'label': 'Knock'}, ...]
            for pred in predictions:
                label = pred['label']
                score = pred['score']
                
                if label in self.class_mapping and score >= self.confidence_threshold:
                    event_type = self.class_mapping[label]
                    self._handle_detection(event_type, score)
                    # We only process the highest scoring valid class to prevent 
                    # multiple events from the same window. The pipeline returns sorted results.
                    break
                    
        except Exception as e:
            logger.error(f"Ambient inference failed: {e}")
            
    def _handle_detection(self, event_type: str, confidence: float):
        current_time = self.clock.time()
        last_time = self.last_event_times.get(event_type, 0.0)
        
        if (current_time - last_time) >= self.event_cooldown_seconds:
            # Emit Event
            event = AmbientEvent(
                event_type=event_type,
                confidence=confidence,
                timestamp=current_time,
                duration_s=self.window_seconds
            )
            self.last_event_times[event_type] = current_time
            logger.info(f"[AMBIENT EVENT] {event.event_type} detected (confidence: {event.confidence:.2f})")
            
            # FUTURE: deliver to awareness layer here
            # e.g., self.event_bus.publish(event)
            
    def _run_loop(self):
        # 1. Initialize classifier in the worker thread to prevent blocking main thread
        self._init_classifier()
        
        if not self.classifier_ready:
            logger.warning("AmbientAwarenessService worker terminating due to classifier initialization failure.")
            return

        window_samples = int(self.sample_rate * self.window_seconds)
        hop_samples = int(self.sample_rate * self.hop_seconds)
        
        audio_buffer = np.array([], dtype=np.float32)

        while not self.shutdown_event.is_set():
            try:
                # Wait for chunks from the ambient fan-out queue
                timestamp, chunk = self.audio_bus.get_ambient_chunk(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # If we have enough for a full hop
                if len(audio_buffer) >= hop_samples:
                    # Extract the most recent window_samples for classification
                    window_to_process = audio_buffer[-window_samples:]
                    
                    # Process window
                    self._process_audio_window(window_to_process)
                    
                    # Slide the window by hop_size
                    audio_buffer = audio_buffer[hop_samples:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ambient worker exception: {e}")
                
        logger.info("AmbientAwarenessService worker stopped cleanly.")
