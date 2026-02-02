import os
import sys
import torch
import clip
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gan_manager import GANManager

def run_miner():
    # --- CONFIGURAZIONE ---
    MODEL_PATH = './resources/network-snapshot-000280.pkl'
    OUTPUT_PATH = './data/dataset_clip.pkl'
    NUM_SAMPLES = 5000  # Numero di immagini da analizzare
    
    # Definizione dei "mood" estetici (Prompt)
    PROMPTS = [
        "minimalist geometric abstract pattern",
        "organic fluid biological shapes",
        "dark moody aggressive industrial glitch",
        "bright vibrant psychedelic energy"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilizzando dispositivo: {device}")

    # 1. Carica CLIP
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(PROMPTS).to(device)

    # 2. Carica GAN
    gan = GANManager(MODEL_PATH, use_gpu=True)
    
    dataset = []

    print(f"Inizio mining di {NUM_SAMPLES} campioni...")
    for i in tqdm(range(NUM_SAMPLES)):
        # Genera vettore Z casuale
        z = torch.randn(1, gan.latent_dim).to(device)
        
        # Genera immagine 
        with torch.no_grad():
            
            img_np = gan.generate_image(np.zeros(1024), None) # Passiamo audio nullo
            img_pil = Image.fromarray(img_np)
            
            # Pre-processa per CLIP
            image_input = preprocess(img_pil).unsqueeze(0).to(device)
            
            # Calcola similarità con i prompt
            logits_per_image, _ = model_clip(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Salva coppia [Z, Scores]
            dataset.append({
                'z': z.cpu().numpy(),
                'scores': probs
            })

    # Salva il dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({'prompts': PROMPTS, 'data': dataset}, f)
    
    print(f"Mining completato! Dataset salvato in {OUTPUT_PATH}")

if __name__ == "__main__":
    run_miner()