import numpy as np
import librosa
from dataclasses import dataclass

@dataclass
class AudioAnalysis:
    spectral_contrast: float
    spectral_flatness: float
    onset_strength: float
    zero_crossing_rate: float
    chroma_variance: float

class AudioFeatureExtractor:
    def process_audio(self, audio_data: np.ndarray, sr: int) -> AudioAnalysis:
        if len(audio_data) == 0:
            return self._get_empty_features()
            
        y = audio_data.astype(np.float32)
        n_samples = len(y)
        
        # --- 1. FILTRO SILENZIO (Cruciale!) ---
        # Calcoliamo l'RMS prima di tutto.
        # Se il chunk è silenzioso (sotto -60dB), le feature spettrali sono spazzatura.
        rms_val = np.sqrt(np.mean(y**2))
        if rms_val < 0.005: # Soglia empirica per il silenzio
            return self._get_empty_features()

        # Configurazione FFT sicura
        n_fft = min(n_samples, 2048)
        if n_fft < 128: # Troppo piccolo per analizzare frequenze
            return self._get_empty_features()
        hop_length = n_fft // 4

        try:
            # --- SPECTRAL CONTRAST (Migliorato) ---
            # band=6 crea 7 bande.
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=6)
            # PRENDIAMO SOLO LE BANDE MEDIE-ALTE
            # contrast[0] sono i bassi profondi (spesso piatti). 
            # Facciamo la media da contrast[1] in su per vedere la vera definizione del suono.
            avg_contrast = float(np.mean(contrast[1:, :]))

            # --- SPECTRAL FLATNESS ---
            # Aggiungiamo un piccolissimo rumore (amin) per evitare log(0) e valori infiniti
            flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length, amin=1e-10)
            avg_flatness = float(np.mean(flatness))

            # --- ALTRE FEATURE ---
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            avg_onset = float(np.mean(onset_env))

            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            avg_zcr = float(np.mean(zcr))

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, tuning=0.0)
            chroma_var = float(np.mean(np.var(chroma, axis=1)))

        except Exception:
            return self._get_empty_features()

        return AudioAnalysis(
            spectral_contrast=round(avg_contrast, 2),
            spectral_flatness=round(avg_flatness, 4),
            onset_strength=round(avg_onset, 3),
            zero_crossing_rate=round(avg_zcr, 4),
            chroma_variance=round(chroma_var, 4)
        )

    def _get_empty_features(self):
        # Ritorniamo None o valori negativi per indicare "Dato non valido"
        # Così possiamo filtrarli dopo nel dataset generation
        return AudioAnalysis(-1.0, -1.0, -1.0, -1.0, -1.0)