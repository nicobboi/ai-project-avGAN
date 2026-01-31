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
        if np.sqrt(np.mean(y**2)) < 0.005: return self._get_empty_features() #calcolo dell'energia media del segnale, sotto la soglia restituisce valori vuoti per evitare movimenti di immagine non richiesti

        try:
            # Analisi a 1024 FFT
            S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) #ottimizzazione, trasformata di FOurier eseguita una sola volta per tutto il chunk
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=4) # Analisi differenza tra picchi di energia e valli dello spettro, permette di capire il livello di definizione del suono
            flatness = librosa.feature.spectral_flatness(S=S) # Aggiungiamo un piccolissimo rumore (amin) per evitare log(0) e valori infiniti
            #------- Altre feature -------
            onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)
            chroma = librosa.feature.chroma_stft(S=S, sr=sr, tuning=0.0)

            return AudioAnalysis(
                round(float(np.mean(contrast[1:, :])), 2),
                round(float(np.mean(flatness)), 4),
                round(float(np.mean(onset_env)), 3),
                round(float(np.mean(zcr)), 4),
                round(float(np.mean(np.var(chroma, axis=1))), 4)
            )
        except: return self._get_empty_features()

    def _get_empty_features(self):
        return AudioAnalysis(-1.0, -1.0, -1.0, -1.0, -1.0) 
    # Ritorniamo None o valori negativi per indicare "Dato non valido"
        # Così possiamo filtrarli dopo nel dataset generation