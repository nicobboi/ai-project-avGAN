import numpy as np
import librosa
from dataclasses import dataclass
import utils.logutils as log

@dataclass
class AudioAnalysis:
    """
    Contenitore dati per le feature musicali.
    """
    loudness_db: float  # Volume in decibel (negativo)
    danceability: float # Indice di variazione ritmica (0.0 - 10.0+)
    brightness: float   # Frequenza media (Hz) - 'Colore' del suono
    energy: float       # RMS medio (0.0 - 1.0)

class AudioFeatureExtractor:
    def __init__(self):
        # Librosa è stateless, non richiede inizializzazione di algoritmi
        pass

    def process_audio(self, audio_data: np.ndarray, sr: int) -> AudioAnalysis:
        """
        Estrae feature musicali usando Librosa.
        Input: Array numpy float32, Sample Rate.
        """
        
        # 1. Controllo validità dati
        if len(audio_data) == 0:
            return self._get_empty_features()
            
        # Assicuriamoci che sia float32 per le performance
        y = audio_data.astype(np.float32)
        n_samples = len(y)

        # --- CONFIGURAZIONE DINAMICA FFT ---
        # Librosa di default usa n_fft=2048. Se il chunk è più piccolo (es. 1024), crasha.
        # Qui calcoliamo un n_fft che stia sempre dentro l'array.
        
        # Se i campioni sono meno di 2048, usa tutta la lunghezza disponibile.
        # Altrimenti usa il classico 2048.
        n_fft = min(n_samples, 2048)
        
        # Hop length: passo di avanzamento. Di standard è n_fft / 4.
        hop_length = n_fft // 4
        
        # Caso limite: se il chunk è minuscolo (es. < 32 sample), impossibile analizzare.
        if n_fft < 32:
            return self._get_empty_features()

        # 2. ESTRAZIONE FEATURE
        try:
            # "Danceability" (Onset Strength)
            # Passiamo n_fft e hop_length calcolati dinamicamente
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            danceability = float(np.std(onset_env))

            # RMS (Energia)
            # Nota: rms usa il parametro 'frame_length' invece di 'n_fft', ma è lo stesso concetto
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
            avg_energy = float(np.mean(rms))
            
            # Loudness in dB
            loudness_db = float(librosa.amplitude_to_db(np.array([avg_energy]))[0])

            # Spectral Centroid (Luminosità)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            avg_brightness = float(np.mean(cent))

        except Exception as e:
            # Se qualcosa fallisce nei calcoli matematici (es. divisione per zero), 
            # restituisci valori vuoti sicuri
            log.error(f"Errore calcolo feature: {e}")
            return self._get_empty_features()

        return AudioAnalysis(
            loudness_db=round(loudness_db, 2),
            danceability=round(danceability, 3),
            brightness=round(avg_brightness, 2),
            energy=round(avg_energy, 4)
        )

    def _get_empty_features(self):
        """Restituisce un oggetto vuoto in caso di errore/silenzio"""
        return AudioAnalysis(-80.0, 0.0, 0.0, 0.0)