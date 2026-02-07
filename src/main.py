import os
import sys

# FIX DLL ERROR: Import torch before everything else
try:
    import torch
except:
    pass

import time
import threading
import queue
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtMultimedia import QMediaPlayer
import OpenGL.GL as GL
import SpoutGL

from gan_manager import GANManager
from mlp_manager import MoodPredictor
from audio_manager import AudioManager
from gui import GUI
from audio_feature_extractor import AudioFeatureExtractor
import utils.logutils as log

### ENVIRONMENT VARIABLES ###
USE_GPU = True
MLP_MODEL_PATH = './resources/mood_mlp.pth'
MLP_SCALER_PATH = './resources/scaler.pkl'
GAN_MODEL_PATH = './resources/network-snapshot-000280.pkl'
SPOUT_SENDER_NAME = "GAN_Visualizer_TD"

def _log_env_variables():
    log.info("Environment Variables:")
    log.info(f"USE_GPU: {USE_GPU}")
    log.info(f"MLP_MODEL_PATH: {MLP_MODEL_PATH}")
    log.info(f"MLP_SCALER_PATH: {MLP_SCALER_PATH}")
    log.info(f"GAN_MODEL_PATH: {GAN_MODEL_PATH}")
    log.info(f"SPOUT_SENDER_NAME: {SPOUT_SENDER_NAME}")

##############################


class VisualizerApp:
    """
    Main application controller.
    
    Orchestrates the interaction between Audio capture, Feature Extraction, 
    MLP prediction, and GAN image generation. Manages threading to keep 
    audio analysis and image synthesis separate from the UI loop.
    """
    def __init__(self):
        """
        Initializes the application components, loads models, and starts background threads.
        """
        self.running = True
        
        # Target: 30 FPS for GAN generation
        self.TARGET_GAN_FPS = 30
        self.GAN_FRAME_TIME = 1.0 / self.TARGET_GAN_FPS

        # Initialize management components
        self.extractor = AudioFeatureExtractor()
        self.audio_system = AudioManager(self.extractor)
        self.window = GUI(self.audio_system, img_size=256)
        self.mlp = MoodPredictor(MLP_MODEL_PATH, MLP_SCALER_PATH, use_gpu=USE_GPU)
        self.gan = GANManager(GAN_MODEL_PATH, use_gpu=USE_GPU)
        
        self.frame_queue = queue.Queue(maxsize=1) # Queue for frames ready for the GAN
        self.shared_data = {"chunk": np.zeros(1024), "latent": None, "feats": self.extractor._get_empty_features()} # Shared memory between threads
        
        # Spout configuration
        self.spout = SpoutGL.SpoutSender()
        self.spout.setSenderName(SPOUT_SENDER_NAME)

        # UI at 60Hz (16ms) to avoid micro-lag when retrieving frames
        self.timer = QTimer()
        self.timer.timeout.connect(self._ui_loop)
        self.timer.start(16)
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.window.show()
        
        # START THREADS ONLY AS THE LAST OPERATION
        threading.Thread(target=self._thread_audio, daemon=True).start()
        threading.Thread(target=self._thread_gan, daemon=True).start()

    def _thread_audio(self):
        """
        Background thread for audio processing.
        
        Continuously captures audio chunks, extracts audio features, 
        and uses the MLP to predict the corresponding latent vector for the GAN.
        """
        while self.running:
            if self.audio_system.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                chunk = self.audio_system.get_current_chunk(window_size=1024) # Get audio part from player
                feats = self.extractor.process_audio(chunk, self.audio_system.sample_rate) # Extract audio features
                latent = self.mlp.get_latent_vector(feats.spectral_contrast, feats.spectral_flatness, feats.onset_strength, feats.zero_crossing_rate, feats.chroma_variance) # MLP translates audio data into GAN coordinates
                self.shared_data.update({"chunk": chunk, "latent": latent, "feats": feats}) # Update data that the GAN reads in its thread
            time.sleep(0.01)

    def _thread_gan(self):
        """
        Background thread for GAN generation, limited to 30 FPS.
        
        Reads the latest latent vector from shared data and generates an image.
        """
        while self.running:
            t_start = time.time()
            if self.shared_data["latent"] is not None:
                img = self.gan.generate_image(self.shared_data["chunk"], self.shared_data["latent"])
                if not self.frame_queue.full():
                    self.frame_queue.put(img)
            
            # Precise FPS limiter
            elapsed = time.time() - t_start
            sleep_needed = self.GAN_FRAME_TIME - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    def _ui_loop(self):
        """
        Main Thread: Log, GUI updates, and Video sending.
        
        Retrieves the generated frame from the queue, updates the PyQt window,
        sends the texture via Spout, and prints debug info to the console.
        """
        f = self.shared_data["feats"]
        sys.stdout.write(f"\r[NORM] C:{f.spectral_contrast:.2f} F:{f.spectral_flatness:.4f} O:{f.onset_strength:.3f} Z:{f.zero_crossing_rate:.4f} CV:{f.chroma_variance:.4f} | GAN Move:{self.gan.get_distance_to_target():.2f}\n")
        sys.stdout.flush()

        try:
            # If a frame is ready, take it immediately (thanks to the 60Hz UI timer)
            img = self.frame_queue.get_nowait()
            self.frame_count += 1
            self.window.set_image(img)
            self.spout.sendImage(img.tobytes(), img.shape[1], img.shape[0], GL.GL_RGB, False, 0)
        except queue.Empty:
            pass
        
        # Calculate real FPS displayed every second
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.window.update_fps_label(self.frame_count, self.TARGET_GAN_FPS)
            self.frame_count = 0
            self.last_fps_time = now

    def __del__(self):
        """
        Destructor to ensure threads are stopped cleanly.
        """
        self.running = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _log_env_variables()
    core = VisualizerApp()
    sys.exit(app.exec())