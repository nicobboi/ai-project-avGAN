import numpy as np
from pydub import AudioSegment
import threading
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, QObject, pyqtSignal

import utils.logutils as log


class AudioManager(QObject):
    # Segnale PUBBLICO: Emesso quando tutto è pronto (per la GUI)
    decoding_finished = pyqtSignal()

    # Segnale INTERNO: Usato per passare i dati dal thread di calcolo decodifica al Main Thread
    # Trasporta: (Array Numpy dei dati, Sample Rate intero)
    _internal_data_ready = pyqtSignal(object, int)

    def __init__(self):
        super().__init__()
        
        # Setup MediaPlayer -> riproduzione audio
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        # Setup dati per l'IA
        self.full_audio_data = np.array([], dtype=np.float32)
        self.sample_rate = 44100
        
        # Connettiamo il segnale interno alla funzione che salva i dati
        self._internal_data_ready.connect(self._finalize_loading)

    def load_file(self, file_path_str):
        """
        Caricamento del file
        - Configura il player.
        - Avvia la decodifica in background.
        """
        # Setup Player Qt
        self.player.setSource(QUrl.fromLocalFile(file_path_str))
        
        # Reset dati vecchi
        self.full_audio_data = np.array([], dtype=np.float32)
        
        # Avvia thread per decodifica file audio
        worker_thread = threading.Thread(target=self._pydub_worker, args=(file_path_str,))
        worker_thread.daemon = True 
        worker_thread.start()

    def _pydub_worker(self, file_path):
        """
        Decodifica file audio con PyDub.
        """
        try:
            log.info(f"Caricamento file: {file_path}...")
            
            # Carica audio con PyDub
            audio = AudioSegment.from_wav(file_path)

            # Ottieni info
            sr = audio.frame_rate
            # Conversione in mono (se stereo)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Conversione in array NumPy (Int16)
            samples = np.array(audio.get_array_of_samples())

            # --- CALCOLO AUTOMATICO NORMALIZZAZIONE ---
            # audio.sample_width ti dice quanti BYTES usa (2=16bit, 3=24bit, 4=32bit)
            bytes_per_sample = audio.sample_width
            bits_per_sample = bytes_per_sample * 8
            
            # Calcoliamo il divisore usando l'operatore bitwise shift
            # Esempio 16 bit: 1 << 15 = 32768
            # Esempio 24 bit: 1 << 23 = 8388608
            max_val = float(1 << (bits_per_sample - 1))
            
            # Normalizzazione in Float32 (-1.0 a 1.0) per la GAN
            y = np.clip(samples.astype(np.float32) / max_val, -1.0, 1.0)
            
            # EMETTI IL SEGNALE
            self._internal_data_ready.emit(y, sr)
            
        except Exception as e:
            log.error(f"Errore caricamento: {e}")

    def _finalize_loading(self, data, sr):
        """
        Emette evento di termine caricamento e decodifica file.
        """
        self.full_audio_data = data
        self.sample_rate = sr
        log.success(f"Dati Audio Pronti! Campioni: {len(data)}, SR: {sr}")
        self.decoding_finished.emit()

    def get_current_chunk(self, window_size=1024):
        """
        Restituisce la porzione di audio corrispondente al tempo attuale del player.
        """
        if len(self.full_audio_data) == 0:
            return np.zeros(window_size, dtype=np.float32)

        # Posizione in millisecondi dal player
        current_ms = self.player.position()
        
        # Conversione in indice array (Secondi * Campioni_al_Secondo)
        idx = int((current_ms / 1000.0) * self.sample_rate)
        
        if idx + window_size > len(self.full_audio_data):
            padding = np.zeros(window_size, dtype=np.float32)
            return padding
            
        return self.full_audio_data[idx : idx + window_size]

    def play_pause(self):
        """Gestisce il toggle Play/Pausa."""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return False # Ritorna False se è in pausa
        else:
            self.player.play()
            return True # Ritorna True se sta suonando

    def stop(self):
        """Ferma la riproduzione."""
        self.player.stop()

    def set_position(self, position_ms):
        """Sposta la riproduzione a un punto specifico (in millisecondi)."""
        self.player.setPosition(position_ms)

    def get_duration(self):
        """Ritorna la durata totale in millisecondi."""
        return self.player.duration()