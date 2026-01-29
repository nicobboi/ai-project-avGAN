import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import utils.logutils as log

class MoodLatentMLP(nn.Module):
    def __init__(self, input_size=4, output_size=512):
        super(MoodLatentMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.network(x)

class MoodPredictor:
    def __init__(self, model_path, scaler_path, use_gpu=True):
        """
        Inizializza il predittore caricando modello e scaler.
        """

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            log.info("MLP Manager: Modalità GPU (CUDA) attivata.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                log.warning("MLP Manager: GPU richiesta ma non trovata. Fallback su CPU.")
            else:
                log.info("MLP Manager: Modalità CPU forzata.")

        self.model_loaded = False
        
        # Controllo esistenza file
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            log.error(f"ERRORE: File modello o scaler non trovati:\nModello: {model_path}\nScaler: {scaler_path}")
            return

        try:
            # Caricamento Scaler
            self.scaler = joblib.load(scaler_path)
            
            # Caricamento Modello
            self.model = MoodLatentMLP(input_size=4, output_size=512)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            log.success(f"MoodPredictor inizializzato su {self.device}.")
            
        except Exception as e:
            log.error(f"ERRORE durante il caricamento del modello: {e}")
            self.model_loaded = False

    def get_latent_vector(self, loudness: float, danceability: float, brightness: float, energy: float):
        """
        Prende le feature audio grezze, le normalizza e restituisce il vettore latente.
        
        Input: 4 float
        Output: Numpy array (512,) float32
        """
        if not self.model_loaded:
            # Ritorna vettore vuoto o zeri in caso di errore
            return np.zeros(512, dtype=np.float32)

        # Preparazione Input (shape [1, 4])
        # L'input deve essere un array 2D per lo scaler e per PyTorch
        raw_input = np.array([[loudness, danceability, brightness, energy]], dtype=np.float32)

        # Normalizzazione (Usando lo scaler appreso nel training)
        try:
            scaled_input = self.scaler.transform(raw_input)
        except Exception as e:
            log.error(f"Errore nello scaling: {e}")
            return np.zeros(512, dtype=np.float32)

        # Conversione in Tensore
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)

        # Inferenza
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Conversione Output in Numpy
        latent_vector = output_tensor.cpu().numpy().flatten()

        return latent_vector