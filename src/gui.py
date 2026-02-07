import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSlider, QStyle)
from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtGui import QImage, QPixmap
import numpy as np

class GUI(QWidget):
    """
    Main Graphical User Interface for the Audio Visualizer.
    
    Handles the display of the generated GAN imagery, audio playback controls,
    and performance metrics (FPS).
    """
    def __init__(self, audio_manager, img_size=256):
        super().__init__()
        self.audio = audio_manager
        
        # Flag to avoid conflicts when the user drags the slider
        self.user_is_seeking = False 
        
        self.setup_ui(img_size=img_size)
        self.connect_signals()

    def setup_ui(self, img_size=256):
        """
        Initializes the widget layout, styles, and UI components.
        """
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

        self.lbl_fps = QLabel("FPS: -- / --")
        self.lbl_fps.setStyleSheet("color: #888; font-size: 10pt;") 
        self.lbl_fps.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.lbl_fps)

        # File Info
        self.lbl_title = QLabel("Select an audio file...")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_title)

        self.lbl_image = QLabel("Waiting for audio...")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(img_size, img_size)
        layout.addWidget(self.lbl_image)

        # Progress Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)

        # Time Label (e.g., 00:00 / 03:45)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.lbl_time)

        # Controls
        controls = QHBoxLayout()
        
        self.btn_open = QPushButton("📂 Open")
        self.btn_open.clicked.connect(self.open_file_dialog)
        
        # Standard system icons for Play/Stop
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
        """
        Connects UI widgets (slider, buttons) and Audio Player signals 
        to their respective slot methods.
        """
        # Slider Events (User drags)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.valueChanged.connect(self.on_slider_move)

        # Player Events (Audio advances)
        # Accessing the QMediaPlayer object inside audio_manager
        self.audio.player.positionChanged.connect(self.update_position)
        self.audio.player.durationChanged.connect(self.update_duration)
        self.audio.player.mediaStatusChanged.connect(self.handle_media_status)

    ### GUI Logic ###

    def set_image(self, img_array: np.ndarray):
        """
        Updates the main label with a new image frame from the GAN.
        
        Args:
            img_array (np.ndarray): The image data in RGB format.
        """
        if img_array is None:
            return
        
        # The gan_manager returns an image with shape (H, W, 3) and type uint8
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        
        # Convert the numpy array to QImage (RGB888 format)
        q_img = QImage(img_array.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Apply the image to the Label
        self.lbl_image.setPixmap(QPixmap.fromImage(q_img))

    def open_file_dialog(self):
        """
        Opens a system file dialog to select a WAV file and initiates loading.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio (*.wav)")
        if file_path:
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.slider.setEnabled(False)
            self.lbl_title.setText(f"Decoding in progress: {os.path.basename(file_path)}...")

            try:
                self.audio.decoding_finished.disconnect()
            except TypeError:
                pass

            self.audio.decoding_finished.connect(lambda: self.on_decoding_complete(file_path))
            self.audio.load_file(file_path)

    def on_decoding_complete(self, file_path):
        """Slot called when the AudioManager has finished processing the file."""
        self.lbl_title.setText(os.path.basename(file_path))
        self.start_audio()

    def start_audio(self):
        """
        Prepares the UI controls for playback and starts the audio.
        """
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.slider.setEnabled(True)
        self.slider.setValue(0)
        self.toggle_play()

    def toggle_play(self):
        """
        Toggles between Play and Pause states and updates the button icon.
        """
        is_playing = self.audio.play_pause()
        icon = QStyle.StandardPixmap.SP_MediaPause if is_playing else QStyle.StandardPixmap.SP_MediaPlay
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def stop_audio(self):
        """
        Stops playback and resets the UI to the initial state.
        """
        self.audio.stop()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.slider.setValue(0)

    # --- FPS Logic ---

    def update_fps_label(self, actual_fps, target_fps):
        """
        Updates the FPS label with conditional colors.
        
        Args:
            actual_fps (float): The current calculated FPS.
            target_fps (float): The target FPS (e.g., 30 or 60).
        """
        
        # Color it red if performance drops too much (below 80% of target)
        color = "#eee"
        if target_fps > 0 and actual_fps < (target_fps * 0.8):
            color = "#ff5555"
            
        self.lbl_fps.setStyleSheet(f"color: {color}; font-size: 10pt; font-weight: bold;")
        self.lbl_fps.setText(f"FPS: {actual_fps} / {target_fps}")

    # --- Slider and Time Logic ---

    def update_duration(self, duration):
        """Sets the maximum value of the slider based on audio duration."""
        self.slider.setMaximum(duration)

    def update_position(self, position):
        """Updates the slider position and time label as audio plays."""
        # Update the slider only if the user is NOT dragging it
        if not self.user_is_seeking:
            self.slider.setValue(position)
        
        self.update_time_label(position, self.audio.get_duration())

    def on_slider_pressed(self):
        """Called when the user clicks/holds the slider."""
        self.user_is_seeking = True

    def on_slider_released(self):
        """Called when the user releases the slider."""
        # When the user releases the slider, update the audio position
        self.audio.set_position(self.slider.value())
        self.user_is_seeking = False

    def on_slider_move(self):
        """Called whenever the slider value changes (used during dragging)."""
        # Update the time label while dragging, even if audio hasn't jumped there yet
        if self.user_is_seeking:
            self.update_time_label(self.slider.value(), self.audio.get_duration())

    def update_time_label(self, current_ms, total_ms):
        """
        Formats milliseconds into MM:SS format and updates the label.
        """
        def format_time(ms):
            seconds = (ms // 1000) % 60
            minutes = (ms // 60000)
            return f"{minutes:02}:{seconds:02}"
        
        self.lbl_time.setText(f"{format_time(current_ms)} / {format_time(total_ms)}")

    def handle_media_status(self, status):
        """
        Handles media status changes (e.g., stopping when the file ends).
        """
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.stop_audio()