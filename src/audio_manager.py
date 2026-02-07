import numpy as np
import librosa
import threading
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, QObject, pyqtSignal

import utils.logutils as log


class AudioManager(QObject):
    """
    Manages audio playback and raw data handling for analysis.
    
    This class bridges the gap between the UI's media player (QtMultimedia)
    and the underlying numpy data needed for the AI/Feature Extraction.
    It handles file loading in a background thread to prevent UI freezing.
    """

    # PUBLIC Signal: Emitted when everything is ready (for the GUI)
    decoding_finished = pyqtSignal()

    # INTERNAL Signal: Used to pass data from the decoding thread to the Main Thread
    # Carries: (Numpy Array of data, Integer Sample Rate)
    _internal_data_ready = pyqtSignal(object, int)

    def __init__(self, audio_feature_extractor=None):
        """
        Initializes the media player and audio data structures.
        """
        super().__init__()
        
        # Setup MediaPlayer -> audio playback
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        # Setup data for AI
        self.full_audio_data = np.array([], dtype=np.float32)
        self.sample_rate = 44100
        self.audio_feature_extractor = audio_feature_extractor
        
        # Connect internal signal to the data saving function
        self._internal_data_ready.connect(self._finalize_loading)

    def load_file(self, file_path_str):
        """
        Loads an audio file.
        - Configures the Qt player.
        - Starts decoding in the background.
        
        Args:
            file_path_str (str): The absolute path to the audio file.
        """
        # Qt Player Setup
        self.player.setSource(QUrl.fromLocalFile(file_path_str))
        
        # Reset old data
        self.full_audio_data = np.array([], dtype=np.float32)
        
        # Start thread for audio file decoding
        worker_thread = threading.Thread(target=self._decode_audio, args=(file_path_str,))
        worker_thread.daemon = True 
        worker_thread.start()

    def _decode_audio(self, file_path):
        """
        Decodes audio file (Worker Thread).
        """
        try:
            log.info(f"Loading file: {file_path}...")
            
            # Load audio with PyDub (Commented out legacy code)
            #audio = AudioSegment.from_wav(file_path)
            #
            ## Get info
            #sr = audio.frame_rate
            ## Convert to mono (if stereo)
            #if audio.channels > 1:
            #    audio = audio.set_channels(1)
            #
            ## Convert to NumPy array (Int16)
            #samples = np.array(audio.get_array_of_samples())
            #
            ## --- AUTOMATIC NORMALIZATION CALCULATION ---
            ## audio.sample_width tells you how many BYTES it uses (2=16bit, 3=24bit, 4=32bit)
            #bytes_per_sample = audio.sample_width
            #bits_per_sample = bytes_per_sample * 8
            #
            ## Calculate the divisor using bitwise shift operator
            ## Example 16 bit: 1 << 15 = 32768
            ## Example 24 bit: 1 << 23 = 8388608
            #max_val = float(1 << (bits_per_sample - 1))
            #
            ## Normalization to Float32 (-1.0 to 1.0) for the GAN
            #y = np.clip(samples.astype(np.float32) / max_val, -1.0, 1.0)
            
            # Actual loading using Librosa
            y, sr = librosa.load(file_path, sr=44100, mono=True)
            
            # EMIT THE SIGNAL
            self._internal_data_ready.emit(y, sr)
            
        except Exception as e:
            log.error(f"Loading error: {e}")

    def _finalize_loading(self, data, sr):
        """
        Slot called when the background thread finishes decoding.
        
        Stores the raw data, triggers feature pre-computation, and notifies the UI.
        """
        self.full_audio_data = data
        if self.audio_feature_extractor is not None:
            self.audio_feature_extractor.compute_features_ranges(data, sr)
        self.sample_rate = sr
        log.success(f"Audio Data Ready! Samples: {len(data)}, SR: {sr}")
        self.decoding_finished.emit()

    def get_current_chunk(self, window_size=1024):
        """
        Returns the portion of audio corresponding to the current player time.
        
        Used to synchronize the visualizer/GAN with the audio being heard.
        
        Args:
            window_size (int): The number of samples to retrieve.
        
        Returns:
            np.array: A float32 array of the audio chunk.
        """
        if len(self.full_audio_data) == 0:
            return np.zeros(window_size, dtype=np.float32)

        # Position in milliseconds from the player
        current_ms = self.player.position()
        
        # Conversion to array index (Seconds * Samples_per_Second)
        idx = int((current_ms / 1000.0) * self.sample_rate)
        
        if idx + window_size > len(self.full_audio_data):
            padding = np.zeros(window_size, dtype=np.float32)
            return padding
            
        return self.full_audio_data[idx : idx + window_size]

    def play_pause(self):
        """
        Handles Play/Pause toggle.
        
        Returns:
            bool: True if playing, False if paused.
        """
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return False # Return False if paused
        else:
            self.player.play()
            return True # Return True if playing

    def stop(self):
        """Stops playback."""
        self.player.stop()

    def set_position(self, position_ms):
        """Moves playback to a specific point (in milliseconds)."""
        self.player.setPosition(position_ms)

    def get_duration(self):
        """Returns the total duration in milliseconds."""
        return self.player.duration()