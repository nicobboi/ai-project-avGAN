from gan_manager import GANManager
from mlp_manager import MoodPredictor
from audio_manager import AudioManager
from gui import GUI
from audio_feature_extractor import AudioFeatureExtractor
from utils.custom_enum import FPS, SampleWindowSize

import utils.logutils as log

import sys
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtMultimedia import QMediaPlayer

import SpoutGL
import OpenGL.GL as GL

### CUSTOM VARIABLES ###

# Set Visualizer FPS
FRAMERATE = FPS.FPS_30
# Set sample window size to retrieve real time from the audio
SAMPLE_WINDOW_SIZE = SampleWindowSize.WS_1024

GAN_MODEL_PATH = './resources/network-snapshot-000280.pkl'
MLP_MODEL_PATH = './resources/mood_mlp.pth'
MLP_SCALAR_PATH = './resources/scaler.pkl'
USE_GPU = False
EVAL_MODE = False

def log_constants():
    log.info("Starting application with the following constants:")
    log.info(f"FRAMERATE: {int(1000 / FRAMERATE)}")
    log.info(f"SAMPLE_WINDOW_SIZE: {SAMPLE_WINDOW_SIZE}")
    log.info(f"GAN_MODEL_PATH: {GAN_MODEL_PATH}")
    log.info(f"MLP_MODEL_PATH: {MLP_MODEL_PATH}")
    log.info(f"MLP_MODEL_PATH: {MLP_SCALAR_PATH}")
    log.info(f"USE_GPU: {USE_GPU}")
    log.info(f"EVAL_MODE: {EVAL_MODE}")

########################

class VisualizerApp:
    def __init__(self):
        # Inizializza audio e gui managers
        self.audio_system = AudioManager()
        self.window = GUI(self.audio_system, img_size=256)

        self.mlp_manager = MoodPredictor(
            model_path=MLP_MODEL_PATH,
            scaler_path=MLP_SCALAR_PATH
        )

        self.gan_manager = GANManager(
            model_path=GAN_MODEL_PATH,
            latent_dim=256,
            use_gpu=USE_GPU
        )

        self.audio_extractor = AudioFeatureExtractor()
        
        self.spout_sender = SpoutGL.SpoutSender()
        self.spout_name = "GAN_Visualizer_TD"
        self.spout_sender.setSenderName(self.spout_name)
        log.info(f"Spout Sender avviato con nome: {self.spout_name}")

        self.frame_count = 0
        self.last_time = time.time()
        self.fps_update_interval = 0.5
        self.fps_timer_acumulator = 0.0
        
        # Configura il timer loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(FRAMERATE)

        # Avvia la GUI
        self.window.show()

    def update_loop(self):
        # Se il player non è in Play (è in Pausa o Fermo), non generare nulla
        if self.audio_system.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return
        
        # --- FPS counter logic ---
        self.frame_count += 1
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        
        self.fps_timer_acumulator += delta_time

        
        if self.fps_timer_acumulator >= self.fps_update_interval:
            actual_fps = int(self.frame_count / self.fps_timer_acumulator)
            
            # Calcolo Target FPS
            target_fps = 1000 / FRAMERATE if FRAMERATE > 0 else 0
            
            self.window.update_fps_label(actual_fps, int(target_fps))
            
            # Reset contatori
            self.frame_count = 0
            self.fps_timer_acumulator = 0.0
        # ------ #

        # Recupero un chunk audio di una finestra temporale
        chunk = self.audio_system.get_current_chunk(window_size=SAMPLE_WINDOW_SIZE)

        # Calcolo feature audio 
        audio_feature = self.audio_extractor.process_audio(chunk, self.audio_system.sample_rate)

        # Generazione vettore latente con MLP
        latent_vector = self.mlp_manager.get_latent_vector(
            loudness=audio_feature.loudness_db,
            danceability=audio_feature.danceability,
            brightness=audio_feature.brightness,
            energy=audio_feature.energy
        )

        # Generazione immagine con GAN
        final_image = self.gan_manager.generate_image(audio_feature.energy,latent_vector)
        if final_image is not None:
            self.window.set_image(final_image)

            height, width, _ = final_image.shape

            # invia l'immagine al canale spout
            self.spout_sender.sendImage(final_image.tobytes(), width, height, GL.GL_RGB, False, 0)

    def __del__(self):
        # Rilascia la memoria di Spout quando l'applicazione si chiude
        if hasattr(self, 'spout_sender'):
            self.spout_sender.releaseSender()


def main():
    app = QApplication(sys.argv)
    log_constants()
    try:
        controller = VisualizerApp()
        sys.exit(app.exec())
    except Exception as e:
        log.error(e)
    finally:
        sys.exit(1)

if __name__ == "__main__":
    main()