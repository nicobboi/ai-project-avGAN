import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- ARCHITETTURA (Sincronizzata con mlp_manager.py) ---
class MoodMLP(nn.Module):
    def __init__(self, input_size=5, output_size=512):
        super(MoodMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x): return self.network(x)

# --- RANGE REALI ---
RANGES = {
    'spectral_contrast':   {'min': (10.0, 13.03), 'med': (12.0, 14.19), 'max': (13.8, 15.33)},
    'spectral_flatness':   {'min': (0.0012, 0.0041), 'med': (0.0028, 0.0090), 'max': (0.0049, 0.0154)},
    'onset_strength':      {'min': (0.207, 0.502), 'med': (0.404, 0.738), 'max': (0.632, 0.869)},
    'zero_crossing_rate':  {'min': (0.0051, 0.0143), 'med': (0.0173, 0.0455), 'max': (0.0375, 0.0666)},
    'chroma_variance':     {'min': (0.0045, 0.0166), 'med': (0.0095, 0.0221), 'max': (0.0126, 0.0267)}
}

def get_feat(name, level):
    low, high = RANGES[name][level]
    return random.uniform(low, high)

def train_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('./data/dataset_clip.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    
    samples = raw_data['data']
    X, Y = [], []

    # Prendiamo le migliori 1000 immagini per ogni categoria
    K = 1000 
    print(f"[*] Creazione dataset bilanciato (Top-{K} per ognuno dei 3 mood)...")
    
    for i in range(len(raw_data['prompts'])):
        category_scores = [s['scores'][i] for s in samples]
        top_indices = np.argsort(category_scores)[-K:]
        
        for idx in top_indices:
            z = samples[idx]['z'].flatten()
            for _ in range(30): # Data Augmentation robusta
                if i == 0: # MOOD 1: ROSSO / AGGRESSIVO
                    feat = [get_feat('spectral_contrast', 'max'), get_feat('spectral_flatness', 'max'), 
                            get_feat('onset_strength', 'max'), get_feat('zero_crossing_rate', 'max'), 
                            get_feat('chroma_variance', 'med')]
                elif i == 1: # MOOD 2: BLU / CALMO
                    feat = [get_feat('spectral_contrast', 'min'), get_feat('spectral_flatness', 'min'), 
                            get_feat('onset_strength', 'min'), get_feat('zero_crossing_rate', 'min'), 
                            get_feat('chroma_variance', 'min')]
                else: # MOOD 3: VIBRANTE (Anche se lo score è basso, prendiamo i migliori)
                    feat = [get_feat('spectral_contrast', 'med'), get_feat('spectral_flatness', 'med'), 
                            get_feat('onset_strength', 'med'), get_feat('zero_crossing_rate', 'med'), 
                            get_feat('chroma_variance', 'max')]
                X.append(feat)
                Y.append(z)

    X_np = np.array(X, dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    Y_np = np.array(Y, dtype=np.float32)

    loader = DataLoader(TensorDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(Y_np)), batch_size=128, shuffle=True)
    model = MoodMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("[*] Training (300 epoche)...")
    for epoch in tqdm(range(300)):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()

    model.to('cpu')
    torch.save(model.state_dict(), './resources/mood_mlp.pth')
    with open('./resources/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[✔] Modello salvato! Ora la GAN reagirà ai 3 colori.")

if __name__ == "__main__":
    train_mlp()