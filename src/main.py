import os
import sys

# FIX DLL ERROR: Import torch prima di tutti
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

class VisualizerApp:
    def __init__(self):
        
        self.running = True
        
        # Target: 30 FPS 
        self.TARGET_GAN_FPS = 30
        self.GAN_FRAME_TIME = 1.0 / self.TARGET_GAN_FPS

        # Inizializzazione componenti gestionali
        self.extractor = AudioFeatureExtractor()
        self.audio_system = AudioManager(self.extractor)
        self.window = GUI(self.audio_system, img_size=256)
        self.mlp = MoodPredictor('./resources/mood_mlp.pth', './resources/scaler.pkl')
        self.gan = GANManager('./resources/network-snapshot-000280.pkl')
        
        self.frame_queue = queue.Queue(maxsize=1) #queue di frame pronti per la GAN
        self.shared_data = {"chunk": np.zeros(1024), "latent": None, "feats": self.extractor._get_empty_features()} #memoria condivisa tra i thread
        
        #configurazione Spout per TD
        self.spout = SpoutGL.SpoutSender()
        self.spout.setSenderName("GAN_Visualizer_TD")

        # UI a 60Hz (16ms) per evitare micro-lag nel recupero frame
        self.timer = QTimer()
        self.timer.timeout.connect(self._ui_loop)
        self.timer.start(16)
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.window.show()
        
        # AVVIA I THREAD SOLO COME ULTIMA OPERAZIONE
        threading.Thread(target=self._thread_audio, daemon=True).start()
        threading.Thread(target=self._thread_gan, daemon=True).start()

    def _thread_audio(self):
        while self.running:
            if self.audio_system.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                chunk = self.audio_system.get_current_chunk(window_size=1024) #prende parte di audio dal player
                feats = self.extractor.process_audio(chunk, self.audio_system.sample_rate) #estrae feature numeriche (es bassi)
                latent = self.mlp.get_latent_vector(feats.spectral_contrast, feats.spectral_flatness, feats.onset_strength, feats.zero_crossing_rate, feats.chroma_variance) #MLP traduce dati audio in coordinate GAN
                self.shared_data.update({"chunk": chunk, "latent": latent, "feats": feats}) #aggiorna i dati che la GAN legge nel suo thread
            time.sleep(0.01)

    def _thread_gan(self):
        """Thread GAN limitato a 30 FPS"""
        while self.running:
            t_start = time.time()
            if self.shared_data["latent"] is not None:
                img = self.gan.generate_image(self.shared_data["chunk"], self.shared_data["latent"])
                if not self.frame_queue.full():
                    self.frame_queue.put(img)
            
            # Limitatore FPS preciso
            elapsed = time.time() - t_start
            sleep_needed = self.GAN_FRAME_TIME - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    def _ui_loop(self):
        """Thread principale: Log, GUI e Invio Video."""
        f = self.shared_data["feats"]
        sys.stdout.write(f"\r[NORM] C:{f.spectral_contrast:.2f} F:{f.spectral_flatness:.4f} O:{f.onset_strength:.3f} Z:{f.zero_crossing_rate:.4f} CV:{f.chroma_variance:.4f} | GAN Move:{self.gan.get_distance_to_target():.2f}   ")
        sys.stdout.flush()

        try:
            # Se c'è un frame pronto, lo prendiamo subito (grazie ai 60Hz della UI)
            img = self.frame_queue.get_nowait()
            self.frame_count += 1
            self.window.set_image(img)
            self.spout.sendImage(img.tobytes(), img.shape[1], img.shape[0], GL.GL_RGB, False, 0)
        except queue.Empty:
            pass
        
        #calcolo FPS reali visualizzati ogni secondo
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.window.update_fps_label(self.frame_count, self.TARGET_GAN_FPS)
            self.frame_count = 0
            self.last_fps_time = now

    def __del__(self):
        self.running = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    core = VisualizerApp()
    sys.exit(app.exec())