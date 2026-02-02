import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class MoodMLP(nn.Module):
    def __init__(self, input_dim=5, output_dim=512):
        super(MoodMLP, self).__init__()
        # La struttura deve riflettere esattamente le "Missing Keys"
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),      # .0
            nn.ReLU(),                      # .1
            nn.BatchNorm1d(64),            # .2 (Running Mean/Var)
            
            nn.Linear(64, 128),            # .3
            nn.ReLU(),                      # .4
            nn.BatchNorm1d(128),            # .5 (Running Mean/Var)
            
            nn.Linear(128, 256),            # .6
            nn.ReLU(),                      # .7
            nn.Linear(256, output_dim)      # .8
        )

    def forward(self, x):
        return self.network(x)

def train_mlp():
    # --- CARICAMENTO DATI ---
    with open('./data/dataset_clip.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    
    prompts = raw_data['prompts']
    samples = raw_data['data']
    
    # --- CREAZIONE DATASET DI TRAINING ---
    # Qui simuliamo l'associazione Audio -> Prompt
    # Esempio: Prompt 0 (Minimal) associato a Spectral Flatness alta
    # Prompt 2 (Aggressive) associato a Onset Strength alta
    X = [] # Feature audio simulate
    Y = [] # Vettori Z corrispondenti
    
    print("Preparazione dataset per l'allenamento...")
    for s in samples:
        z = s['z'].flatten()
        scores = s['scores']
        
        # Se un'immagine ha un punteggio alto (>0.4) per un prompt, 
        # creiamo un esempio di training con feature audio "ideali" per quel mood
        for i, score in enumerate(scores):
            if score > 0.4:
                # Simuliamo 5 feature audio basandoci sul mood i
                if i == 0: audio_feat = [0.8, 0.1, 0.2, 0.1, 0.1] # Minimal
                elif i == 1: audio_feat = [0.4, 0.5, 0.3, 0.4, 0.6] # Organic
                elif i == 2: audio_feat = [0.2, 0.9, 0.9, 0.8, 0.2] # Aggressive
                else: audio_feat = [0.9, 0.6, 0.7, 0.5, 0.9] # Vibrant
                
                X.append(audio_feat)
                Y.append(z)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Converti in Tensori
    X_train = torch.FloatTensor(X_scaled)
    Y_train = torch.FloatTensor(Y)
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- TRAINING ---
    model = MoodMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Inizio allenamento MLP...")
    for epoch in range(100):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")

    # Salva modello e scaler
    torch.save(model.state_dict(), './resources/mood_mlp.pth')
    with open('./resources/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Allenamento completato! Modelli salvati in ./resources/")

if __name__ == "__main__":
    train_mlp()