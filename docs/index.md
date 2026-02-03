## 1. Introduction

### Problem Description

- Motivation for the problem and its relevance. 
<p>
This project addresses the automation of visual content generation in live performance contexts. In audio-visual performances, creating synchronized visuals typically requires manual keyframing or complex node-based programming. The motivation is to leverage generative deep learning to automate this workflow, allowing for real-time visual streams that react coherently to audio features without continuous human intervention.
</p>

-  Target audience interested in the problem.
<p>
<strong>Musicians, Creative Coders, and VJs</strong> interested in integrating automated, AI-driven visual synthesis into their live setups.
</P>

- Benefits of a proposed solution.
<p>
While industry-standard tools like <strong>TouchDesigner</strong> provide extensive control, they have a steep learning curve. The proposed solution offers an abstraction layer: it uses a neural network to handle the mapping between sound and video. This allows users to generate high-quality, semantically meaningful visuals with minimal setup time, acting as a co-creative tool rather than a manual editor.
</p>

### Proposed Solution

- Approach to the solution.
<br>
The system implements a hybrid architecture. <strong>Offline</strong>, we employed <strong>OpenAI CLIP</strong> to mine the GAN's latent space for coordinates matching specific semantic prompts (e.g., "fiery red"), and trained an <strong>MLP</strong> to map audio features to these coordinates. <strong>Real-Time</strong>, the pipeline extracts features via <strong>Librosa</strong>, infers the target latent vector via the MLP, generates frames using <strong>StyleGAN3</strong>, and transmits the output via <strong>SpoutGL</strong>.


- Computational challenges faced.
<br>
We addressed the following engineering challenges:
<br>
&nbsp;&nbsp;+ <strong>Latency vs. Throughput:</strong> StyleGAN3 inference is computationally expensive. To prevent audio dropouts and the loss of FPS, we implemented a <strong>multithreaded architecture.</strong> 
<br>
&nbsp;&nbsp;+ <strong>Synchronization:</strong> Ensuring the visual reaction aligns with audio transients despite inference latency.
<br>
&nbsp;&nbsp;+ <strong>Temporal Coherence:</strong> To avoid jittery transitions, we implemented <strong>Spherical Linear Interpolation (Slerp)</strong> with a dynamic step size modulated by the signal's RMS energy.

- Task distribution within the group.
<br>
The workload was distributed as follows:
<br>
&nbsp;&nbsp;- <strong>Audio Processing:</strong> Nicolò Ragazzi
<br>
&nbsp;&nbsp;- <strong>Model Training & CLIP:</strong> Luigi Botrugno
<br>
&nbsp;&nbsp;- <strong>Generative Engine:</strong> Nicolò Ragazzi, Luigi Botrugno
<br>
&nbsp;&nbsp;- <strong>System Integration:</strong> Nicolò Ragazzi, Luigi Botrugno

- Summary of achieved results.
<br>
We successfully developed a Python application that processes audio input and generates a video stream at a stable <strong>30 FPS</strong>. The system maps 5 distinct audio descriptors to visual parameters and supports interoperability via <strong>Spout</strong>.

## 2. Proposed Method

### Solution Choice

- Alternative solutions considered and justification for the chosen approach.
<br>
We selected <strong>StyleGAN3</strong> over StyleGAN2 due to its <strong>equivariance</strong> properties, which significantly reduce "texture sticking" artifacts during continuous interpolation. We introduced an intermediate <strong>MLP</strong> because mapping raw audio features directly to the latent space ($w$) resulted in noisy output. The MLP serves as a semantic bridge, translating audio features into meaningful latent coordinates.

- Methodology for performance measurement.
<br>
Performance is evaluated quantitatively using a real-time <strong>FPS counter</strong> implemented in the GUI, tracking the effective frame delivery rate via Spout. Qualitatively, we assessed the latency between audio onsets and visual responses during playback.

- Solution's Architecture Details:
<br>
The system relies on three core modules:
<br>
&nbsp;&nbsp;+ <strong>Feature Extraction:</strong> We compute 5 descriptors per 1024-sample chunk:
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Spectral Contrast</em> (Timbre definition)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Spectral Flatness</em> (Noise/Sibilance detection)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Onset Strength</em> (Rhythmic transient detection)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Zero Crossing Rate</em> (Signal roughness)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Chroma Variance</em> (Harmonic complexity)
<br>
&nbsp;&nbsp;+ <strong>Semantic Mapping:</strong> We clustered the latent space using CLIP into three classes:
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Aggressive</em> ("intense fiery red, jagged shapes")
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Calm</em> ("peaceful soft blue, static atmosphere")
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Vibrant</em> ("vibrant multicolored geometric lines")
<br>
&nbsp;&nbsp;+ <strong>Generation:</strong> Transitions are smoothed using energy-dependent interpolation steps to maintain visual fluidity.

## 3. Experimental Results

### Demonstration and Technologies

- Instructions for the demonstration.
<br>
To execute the system:
<br>
&nbsp;&nbsp;* Launch a Spout receiver (e.g., <strong>TouchDesigner</strong>).
<br>
&nbsp;&nbsp;* Run the main script: <code>python main.py</code>.
<br>
&nbsp;&nbsp;* Load a <strong>.wav</strong> file via the GUI.
<br>
&nbsp;&nbsp;* Await the decoding process.
<br>
&nbsp;&nbsp;* Press <strong>Play</strong> to initiate generation and transmission.

- Technologies and versions used (for reproducibility).
<br>
&nbsp;&nbsp;- <strong>Language:</strong> Python 3.12.10
<br>
&nbsp;&nbsp;- <strong>Deep Learning:</strong> PyTorch 2.9.1+cu126 (CUDA required)
<br>
&nbsp;&nbsp;- <strong>DSP:</strong> Librosa 0.11.0, PyDub 0.25.1
<br>
&nbsp;&nbsp;- <strong>Graphics/UI:</strong> PyQt6 6.10.2, SpoutGL, OpenGL
<br>
&nbsp;&nbsp;- <strong>Models:</strong> StyleGAN3 (pretrained snapshot), OpenAI CLIP, Custom MLP.

### Results

- Results of the best configuration.
<br>
Tested on an <strong>NVIDIA RTX 4060</strong>, the system maintains a stable <strong>30 FPS</strong> at <strong>256x256</strong> resolution (upscalable via external post-processing). The perceived latency is <100ms, ensuring effective synchronization. The MLP consistently maps low-frequency energy (bass) to the "Aggressive/Red" cluster as defined in the training phase.

- Ablation Study: Comparison across configurations.
<br>
Ablation testing showed that without the <strong>MLP</strong>, direct mapping of audio to latent space produced incoherent, high-frequency visual noise. Removing the <strong>RMS-based smoothing</strong> resulted in stuttering visuals during low-energy sections and chaotic transitions during high-energy sections.

## 4. Discussion and Conclusions

### Results Discussion

- Analysis of performance in relation to expectations.
<br>
The results align with the initial requirements. The implementation of <strong>asynchronous threading</strong> was critical; it successfully decoupled the heavy GPU inference from the audio playback thread, preventing buffer underruns.

### Method Validity

- Evaluation if the method meets expectations.
<br>
The CLIP-based mining approach proved superior to random selection. It imbues the GAN's output with <strong>semantic reactivity</strong>—changing the visual atmosphere based on audio mood—rather than merely modulating geometric parameters.

### Limitations and Maturity

- Limits of applicability and biases.
<br>
&nbsp;&nbsp;* <strong>Hardware:</strong> Strict dependency on NVIDIA GPUs with <strong>CUDA</strong> support.
<br>
&nbsp;&nbsp;* <strong>Domain Lock:</strong> The system is limited to the visual domain of the pre-trained StyleGAN dataset (e.g., abstract art).
<br>
&nbsp;&nbsp;* <strong>Resolution:</strong> Currently constrained to 256x256px to ensure real-time performance.

### Future Works

- Proposals to advance the project.
<br>
Future development will focus on model optimization using <strong>TensorRT</strong> to increase resolution or framerate. Additionally, integrating a predictive <strong>Beat Tracking</strong> algorithm could compensate for system latency by anticipating musical drops.