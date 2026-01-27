import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSlider, QStyle)
from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtGui import QImage, QPixmap
import numpy as np

class GUI(QWidget):
    def __init__(self, audio_manager, img_size=256):
        super().__init__()
        self.audio = audio_manager
        
        # Flag per evitare conflitti quando l'utente trascina lo slider
        self.user_is_seeking = False 
        
        self.setup_ui(img_size=img_size)
        self.connect_signals()

    def setup_ui(self, img_size=256):
        self.setWindowTitle("Audio Player")
        self.setGeometry(200, 200, 500, 250)
        self.setStyleSheet("""
            QWidget { background-color: #222; color: #eee; font-family: sans-serif; }
            QSlider::groove:horizontal { height: 8px; background: #444; border-radius: 4px; }
            QSlider::handle:horizontal { background: #3498db; width: 16px; margin: -4px 0; border-radius: 8px; }
            QPushButton { padding: 8px; background-color: #444; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
            QPushButton:disabled { color: #777; background-color: #333; }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        # Info file
        self.lbl_title = QLabel("Seleziona un file audio...")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_title)

        self.lbl_image = QLabel("In attesa dell'audio...")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(img_size, img_size)
        layout.addWidget(self.lbl_image)

        # Slider di avanzamento
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)

        # Label tempo (es. 00:00 / 03:45)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.lbl_time)

        # Controlli
        controls = QHBoxLayout()
        
        self.btn_open = QPushButton("📂 Apri")
        self.btn_open.clicked.connect(self.open_file_dialog)
        
        # Icone standard di sistema per Play/Stop
        icon_play = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.btn_play = QPushButton()
        self.btn_play.setIcon(icon_play)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)

        icon_stop = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        self.btn_stop = QPushButton()
        self.btn_stop.setIcon(icon_stop)
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_stop.setEnabled(False)

        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_stop)
        
        layout.addLayout(controls)
        self.setLayout(layout)

    def connect_signals(self):
        # Eventi dello Slider (Utente trascina)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.valueChanged.connect(self.on_slider_move)

        # Eventi dal Player (Audio avanza)
        # Accediamo all'oggetto QMediaPlayer dentro audio_manager
        self.audio.player.positionChanged.connect(self.update_position)
        self.audio.player.durationChanged.connect(self.update_duration)
        self.audio.player.mediaStatusChanged.connect(self.handle_media_status)

    ### Logica GUI ###

    def set_image(self, img_array: np.ndarray):
        if img_array is None:
            return
        
        # Il gan_manager restituisce un'immagine con forma (H, W, 3) e tipo uint8
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        
        # Convertiamo l'array numpy in QImage (formato RGB888)
        q_img = QImage(img_array.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Applichiamo l'immagine alla Label
        self.lbl_image.setPixmap(QPixmap.fromImage(q_img))

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Apri Audio", "", "Audio (*.mp3 *.wav *.ogg *.flac)")
        if file_path:
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.slider.setEnabled(False)
            self.lbl_title.setText(f"Decodifica in corso: {os.path.basename(file_path)}...")

            try:
                self.audio.decoding_finished.disconnect()
            except TypeError:
                pass

            self.audio.decoding_finished.connect(lambda: self.on_decoding_complete(file_path))
            self.audio.load_file(file_path)

    def on_decoding_complete(self, file_path):
        """Slot chiamato quando l'AudioManager ha finito di processare il file."""
        self.lbl_title.setText(os.path.basename(file_path))
        self.start_audio()

    def start_audio(self):
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.slider.setEnabled(True)
        self.slider.setValue(0)
        self.toggle_play()

    def toggle_play(self):
        is_playing = self.audio.play_pause()
        icon = QStyle.StandardPixmap.SP_MediaPause if is_playing else QStyle.StandardPixmap.SP_MediaPlay
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def stop_audio(self):
        self.audio.stop()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.slider.setValue(0)

    # --- Logica Slider e Tempo ---

    def update_duration(self, duration):
        self.slider.setMaximum(duration)

    def update_position(self, position):
        # Aggiorniamo lo slider solo se l'utente NON lo sta trascinando
        if not self.user_is_seeking:
            self.slider.setValue(position)
        
        self.update_time_label(position, self.audio.get_duration())

    def on_slider_pressed(self):
        self.user_is_seeking = True

    def on_slider_released(self):
        # Quando l'utente lascia lo slider, aggiorniamo la posizione audio
        self.audio.set_position(self.slider.value())
        self.user_is_seeking = False

    def on_slider_move(self):
        # Aggiorna l'etichetta del tempo mentre trasciniamo, anche se l'audio non è ancora saltato lì
        if self.user_is_seeking:
            self.update_time_label(self.slider.value(), self.audio.get_duration())

    def update_time_label(self, current_ms, total_ms):
        def format_time(ms):
            seconds = (ms // 1000) % 60
            minutes = (ms // 60000)
            return f"{minutes:02}:{seconds:02}"
        
        self.lbl_time.setText(f"{format_time(current_ms)} / {format_time(total_ms)}")

    def handle_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.stop_audio()