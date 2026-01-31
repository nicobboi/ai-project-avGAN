## 1. Introduction

### Problem Description

- Motivation for the problem and its relevance. 
<p>
Il progetto affronta una sfida nell'ambito video-musicale performativo: rendere più immersivo l'ascolto tramite video visual art generative. Le visual audio sono utilizzate spesso in concerti live o video e possono aumentare l'attenzione dell'ascoltatore e, soprattutto, l'immersione in ciò che sta ascoltando. 
</p>

-  Target audience interested in the problem.
<p>
Musicisti, Video Maker, o comunque chiunque posso essere interessato in video visual art generative audio reactive. 
</P>

- Benefits of a proposed solution.
<p>
Per creare questi tipi di video sono necessarie grandi competenze in software come TouchDesigner ("powerful, node-based visual programming environment designed for creating real-time interactive 2D/3D applications, installations, and multimedia content"), per poter creare design e progetti che siano in grado di poter immergere l'ascoltatore. La soluzione proposta permette di poterne generare facilmente (ma senza avere la presunzione di poter sostituire gli esperti, in quanto la soluzione proposta è difficilmente personalizzabile).
</p>

### Proposed Solution

- Approach to the solution.
<br>
Per ogni frame (obiettivo 30FPS):
1. Analisi chunk audio.
2. Generazione vettore latente da un MLP dato in input un array di samples audio.
3. generazione immagine dalla GAN.
4. condivisione dell'immagine a TouchDesigner per post-processing.


- Computational challenges faced.
<br>
Gestione di un modello pesante come StyleGAN3 in tempo reale; sincronizzazione tra il segnale audio e la generazione video; 
- Task distribution within the group.
- Summary of achieved results.
<br>
Un'applicazione in grado di generare flussi video fluidi a 30 FPS, in grado di reagire a feature audio (come ritmo, armonia) tramite un'architettura asincrona.

## 2. Proposed Method

### Solution Choice

- Alternative solutions considered and justification for the chosen approach.
<br>

Si è valutato l'utilizzo di differenti tipi di reti neurali, ma si è giunti alla scelta della StyleGAN3 per motivi di efficienza, in quanto è in grado di ottimizzare la fluidità delle trasformazioni visive. Inoltre, si è scelto di utilizzare un MLP nella trasmissione audio-GAN, in qusnto ci permette di "predire" una posizione sensata nello spazio latente in base al "mood" trasmesso dalle feature audio. 

- Methodology for performance measurement.

## 3. Experimental Results

### Demonstration and Technologies

- Instructions for the demonstration.
<br>




- Technologies and versions used (for reproducibility).
<br>
Python 3.12.10
<br>
(PyTorch)
<br>
torch == 2.9.1+cu126
torchvision == 0.24.1+cu126
torchaudio == 2.9.1+cu126
<br>
PyQt6 == 6.10.2
pydub == 0.25.1
kornia == 0.8.2
einops == 0.8.1
PyOpenGL == 3.1.10
requests == 2.32.5
scipy == 1.17.0
librosa == 0.11.0
### Results

- Results of the best configuration.
- Ablation Study: Comparison across configurations.
- `[extra]` Comparative Study with Literature[^1].

## 4. Discussion and Conclusions

### Results Discussion

- Analysis of performance in relation to expectations.

### Method Validity

- Evaluation if the method meets expectations.

### Limitations and Maturity

- Limits of applicability and biases.
- Technological maturity of the solution.

### Future Works

- Proposals to advance the project.

[^1]: `[extra]` sections mandatory for groups with more than two members.
