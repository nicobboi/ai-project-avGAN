import os
import sys
import torch
import clip
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# FIX PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.gan_manager import GANManager

def run_miner():
    MODEL_PATH = './resources/network-snapshot-000280.pkl'
    OUTPUT_PATH = './data/dataset_clip.pkl'
    NUM_SAMPLES = 5000 
    
    # DEFINIZIONE 3 MOOD CROMATICI
    PROMPTS = [
        "intense fiery red and orange colors, aggressive sharp jagged shapes, high energy dynamic movement", # ROSSO/INTENSO
        "peaceful soft blue and teal colors, calm static atmosphere, smooth blurry gradients",          # BLU/CALMO
        "vibrant multicolored geometric lines, energetic neon patterns, sharp contrast"                 # INTERMEDIO/LINEE
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Miner avviato su: {device}")

    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(PROMPTS).to(device)
    gan = GANManager(MODEL_PATH, use_gpu=True)
    
    dataset = []

    print(f"[*] Analisi di {NUM_SAMPLES} immagini in corso...")
    for i in tqdm(range(NUM_SAMPLES)):
        z = torch.randn(1, gan.latent_dim).to(device)
        with torch.no_grad():
            img_np = gan.generate_image(np.zeros(1024), None) 
            img_pil = Image.fromarray(img_np)
            
            image_input = preprocess(img_pil).unsqueeze(0).to(device)
            logits_per_image, _ = model_clip(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            dataset.append({'z': z.cpu().numpy(), 'scores': probs})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({'prompts': PROMPTS, 'data': dataset}, f)
    
    print(f"[✔] Mining completato su 3 classi! Salvato in {OUTPUT_PATH}")

if __name__ == "__main__":
    run_miner()