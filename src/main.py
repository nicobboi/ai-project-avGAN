import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtMultimedia import QMediaPlayer

import SpoutGL
import OpenGL.GL as GL

from audio_manager import AudioManager
from gui import GUI
from gan_manager import GANManager
from utils.custom_enum import FPS, SampleWindowSize

import utils.logutils as log

### CUSTOM VARIABLES ###

# Set Visualizer FPS
FRAMERATE = FPS.FPS_30
# Set sample window size to retrieve real time from the audio
SAMPLE_WINDOW_SIZE = SampleWindowSize.WS_1024

MODEL_PATH = './resources/models/light_gan_test/model_5.pt'
USE_GPU = False
EVAL_MODE = False

def log_constants():
    log.info("Starting application with the following constants:")
    log.info(f"FRAMERATE: {FRAMERATE}")
    log.info(f"SAMPLE_WINDOW_SIZE: {SAMPLE_WINDOW_SIZE}")
    log.info(f"MODEL_PATH: {MODEL_PATH}")
    log.info(f"USE_GPU: {USE_GPU}")
    log.info(f"EVAL_MODE: {EVAL_MODE}")

########################

class VisualizerApp:
    def __init__(self):
        # Inizializza audio e gui managers
        self.audio_system = AudioManager()
        self.window = GUI(self.audio_system, img_size=256)

        self.gan_manager = GANManager(
            model_path=MODEL_PATH, 
            image_size=256, 
            latent_dim=256,
            use_gpu=USE_GPU,
            eval_mode=EVAL_MODE
        )
        
        self.spout_sender = SpoutGL.SpoutSender()
        self.spout_name = "GAN_Visualizer_TD"
        self.spout_sender.setSenderName(self.spout_name)
        log.info(f"Spout Sender avviato con nome: {self.spout_name}")
        
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

        # Recupero un chunk audio di una finestra temporale
        chunk = self.audio_system.get_current_chunk(window_size=SAMPLE_WINDOW_SIZE)

        final_image = self.gan_manager.generate_image(chunk)
        if final_image is not None:
            self.window.set_image(final_image)

            height, width, _ = final_image.shape

            # Spout/OpenGL si aspettano un formato RGB. 
            # Se la tua GAN genera BGR (standard OpenCV), de-commenta la riga seguente:
            # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

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