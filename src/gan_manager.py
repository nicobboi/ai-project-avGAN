import torch
import os
import sys
import pickle
import numpy as np
import utils.logutils as log

# IMPORTANTE: Se il pickle richiede moduli specifici di StyleGAN (es. dnnlib),
# assicurati che la cartella stylegan3 sia nel path.
# Se la cartella stylegan3 è nella root del progetto, scommenta la riga sotto:
# sys.path.insert(0, "./stylegan3")

class GANManager:
    def __init__(self, model_path, latent_dim=512, use_gpu=True):
        """
        Nota: image_size in StyleGAN3 è definito nel modello stesso, quindi il parametro 
        passato qui potrebbe essere ignorato a favore di G.img_resolution.
        """
        self.model_path = model_path
        self.latent_dim = latent_dim 
        
        # LOGICA DI SELEZIONE DEVICE
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            log.info("GAN Manager: Modalità GPU (CUDA) attivata.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                log.warning("GAN Manager: GPU richiesta ma non trovata. Fallback su CPU.")
            else:
                log.info("GAN Manager: Modalità CPU forzata.")

        # Carica il modello
        self.model = self._load_model()

        # Aggiorniamo le dimensioni in base al modello caricato
        if hasattr(self.model, 'z_dim'):
            self.latent_dim = self.model.z_dim
        
        # Gestione Classi (c) per StyleGAN
        # Se il modello non è condizionato, c rimane None
        self.c = None
        if hasattr(self.model, 'c_dim') and self.model.c_dim > 0:
            self.c = torch.zeros([1, self.model.c_dim]).to(self.device)

        # Posizione attuale nello spazio (da dove generiamo l'immagine)
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        # Posizione obiettivo verso cui ci stiamo muovendo
        self.target_z = torch.randn(1, self.latent_dim).to(self.device)

    def _load_model(self, eval_mode=True):
        log.info(f"Caricamento modello StyleGAN3 da: {self.model_path}...")

        if not os.path.exists(self.model_path):
            raise FileExistsError(f"Il file del modello non esiste: {self.model_path}")

        try:
            with open(self.model_path, 'rb') as f:
                # Carichiamo il dizionario del network
                network_dict = pickle.load(f)
                
                # Solitamente 'G_ema' è il generatore migliore per inferenza
                if 'G_ema' in network_dict:
                    G = network_dict['G_ema']
                    log.info("Trovato generatore G_ema (Exponential Moving Average).")
                else:
                    G = network_dict['G']
                    log.warning("G_ema non trovato, uso generatore base G.")
                
                G.to(self.device)
                
                # StyleGAN3 è quasi sempre in eval mode per generazione
                G.eval() 
            
            log.success(f"✅ Modello StyleGAN3 caricato! Risoluzione: {G.img_resolution}x{G.img_resolution}")
            return G

        except ModuleNotFoundError as e:
            log.error(f"❌ Errore moduli mancanti: {e}. Assicurati che 'dnnlib' e 'torch_utils' siano visibili a Python.")
            return None
        except Exception as e:
            log.error(f"❌ Errore critico nel caricamento: {e}")
            return None

    def generate_image(self, step_speed=0.05, mlp_latent_vector=None) -> np.uint8:
        """
        Genera un frame GAN combinando:
        1. Navigazione Fluida: Si sposta verso un target (che può essere Random o MLP).
        2. MLP Mood: Se presente, diventa il nuovo target verso cui navigare.
        3. Audio Reactivity: Velocità variabile e "Kick" sugli impulsi.
        """

        with torch.no_grad():
            
            # 1. GESTIONE DEL TARGET (Dove stiamo andando?)
            if mlp_latent_vector is not None:
                # CASO A: C'è audio -> Il target diventa il mood predetto dall'MLP
                # Convertiamo numpy -> tensor
                mlp_target = torch.tensor(mlp_latent_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Normalizziamo il target MLP sulla sfera (fondamentale per evitare artefatti)
                mlp_target = mlp_target / mlp_target.norm() * np.sqrt(self.latent_dim)
                
                # Impostiamo il target verso cui la GAN deve interpolare
                self.target_z = mlp_target
            
            else:
                # CASO B: Silenzio/Nessun Audio -> Il target è un punto random (Random Walk)
                # Se siamo arrivati vicini al vecchio target random, ne scegliamo uno nuovo
                distance_to_target = torch.norm(self.target_z - self.current_z)
                if distance_to_target < 0.2:
                    self.target_z = torch.randn(1, self.latent_dim).to(self.device)

            # 2. CALCOLO VELOCITÀ DI NAVIGAZIONE
            # step_speed arriva dall'esterno (es. basato su RMS volume)
            base_speed = 0.005 
            # Più è alto il volume, più velocemente ci muoviamo verso il target
            dynamic_speed = min(step_speed * 0.1, 0.5) 
            step = base_speed + dynamic_speed

            # 3. INTERPOLAZIONE (La "Navigazione")
            # Spostiamo current_z un pezzettino verso target_z
            # Formula LERP: A = (1-t)*A + t*B
            self.current_z = (1 - step) * self.current_z + step * self.target_z

            # 4. GESTIONE IMPULSI ("Kick") - Opzionale ma consigliato per il ritmo
            # Creiamo una copia temporanea per non sporcare la traiettoria di navigazione
            final_z = self.current_z.clone()
            
            # Se il volume è molto alto (step_speed > 0.5), diamo un colpo extra
            if step_speed > 0.5:
                kick_impact = torch.randn(1, self.latent_dim).to(self.device) * step_speed * 0.2
                final_z += kick_impact

            # 5. NORMALIZZAZIONE FINALE
            # Riporta il vettore sulla superficie dell'ipersfera
            final_z = final_z / final_z.norm() * np.sqrt(self.latent_dim)

            # 6. GENERAZIONE
            img = self.model(final_z, self.c, force_fp32=True, noise_mode='const')

            # --- Formattazione Output ---
            img = img.permute(0, 2, 3, 1)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            final_image = img[0].cpu().numpy()
            
            return final_image