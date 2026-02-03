## 1. Introduction

### Problem Description

- Motivation for the problem and its relevance. 
<p>
The project addresses a specific challenge in performative music visualization: enhancing the listening experience through <strong>generative visual art</strong>. Audio-reactive visuals are frequently used in live concerts and music videos to increase listener engagement and immersion.
</p>

-  Target audience interested in the problem.
<p>
<strong>Musicians</strong>, <strong>Video Makers</strong>, <strong>VJs</strong>, and digital artists interested in real-time, audio-reactive visual content generation.
</P>

- Benefits of a proposed solution.
<p>
Creating high-quality reactive visuals typically requires advanced skills in complex software like <strong>TouchDesigner</strong>. Our solution democratizes this process using <strong>Artificial Intelligence</strong>, allowing users to generate professional-grade, semantically coherent visuals without manual keyframing. While it does not aim to replace expert designers, it offers an accessible tool for automated, high-end visual generation.
</p>

### Proposed Solution

- Approach to the solution.
<br>
The system operates via a hybrid pipeline. In the offline phase, we use <strong>OpenAI CLIP</strong> to explore the GAN's latent space and identify coordinates associated with specific textual prompts (e.g., "calm blue"). An <strong>MLP</strong> is then trained to map audio features to these verified coordinates. In the real-time phase, the application extracts audio features using <strong>Librosa</strong>, predicts the target latent vector via the MLP, generates images using <strong>StyleGAN3</strong>, and streams the video texture to external software via <strong>SpoutGL</strong>.


- Computational challenges faced.
<br>
We successfully addressed several key technical challenges:
<br>
&nbsp;&nbsp;+ <strong>Performance:</strong> Running StyleGAN3 (a heavy model) in real-time was the primary bottleneck. We implemented a <strong>multithreaded architecture</strong> separating audio playback, neural inference, and GUI rendering.
<br>
&nbsp;&nbsp;+ <strong>Synchronization:</strong> Ensuring low-latency alignment between audio onsets and visual changes.
<br>
&nbsp;&nbsp;+ <strong>Fluidity:</strong> To prevent jarring visual jumps, we implemented <strong>Spherical Linear Interpolation (Slerp)</strong>, dynamically modulated by the audio's RMS energy.

- Task distribution within the group.
<br>
The team divided tasks as follows:
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
We developed a Python application capable of loading audio files, analyzing them in real-time, and generating a <strong>30 FPS</strong> video stream reactive to 5 distinct musical features. The output is compatible with professional setups via <strong>Spout</strong> integration.

## 2. Proposed Method

### Solution Choice

- Alternative solutions considered and justification for the chosen approach.
<br>
We selected <strong>StyleGAN3</strong> over other architectures for its superior texture coherence and equivariance, which minimizes "texture sticking" artifacts during animation. Furthermore, integrating an intermediate <strong>MLP</strong> was essential to translate abstract audio "moods" into valid latent coordinates, preventing the chaotic output that would result from driving the GAN directly with raw audio data.

- Methodology for performance measurement.
<br>
Performance is monitored via a real-time <strong>FPS counter</strong> in the GUI, which tracks frames rendered and sent via Spout against the 30 FPS target. We also qualitatively assess the latency between audio transients and visual response.

- Solution's Architecture Details:
<br>
The system relies on three core components:
<br>
&nbsp;&nbsp;+ <strong>Feature Extraction:</strong> We extract 5 descriptors per audio chunk:
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Spectral Contrast</em> (distinguishes tone from noise)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Spectral Flatness</em> (detects noisiness/sibilance)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Onset Strength</em> (detects rhythmic beats)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Zero Crossing Rate</em> (measures signal roughness)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Chroma Variance</em> (measures harmonic complexity)
<br>
&nbsp;&nbsp;+ <strong>Semantic Mapping:</strong> CLIP was used to define three distinct "mood" clusters:
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Aggressive</em> ("intense fiery red, jagged shapes")
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Calm</em> ("peaceful soft blue, static atmosphere")
<br>
&nbsp;&nbsp;&nbsp;&nbsp;- <em>Vibrant</em> ("vibrant multicolored geometric lines")
<br>
&nbsp;&nbsp;+ <strong>Generation:</strong> The transition between latent vectors is smoothed using interpolation steps calculated based on audio intensity.

## 3. Experimental Results

### Demonstration and Technologies

- Instructions for the demonstration.
<br>
To demonstrate the system:
<br>
&nbsp;&nbsp;* Launch <strong>TouchDesigner</strong> (Spout receiver).
<br>
&nbsp;&nbsp;* Run the main script: <code>python main.py</code>.
<br>
&nbsp;&nbsp;* Load a <strong>.wav</strong> file via the GUI.
<br>
&nbsp;&nbsp;* Wait for the decoding phase.
<br>
&nbsp;&nbsp;* Press <strong>Play</strong> to start video generation and Spout transmission.

- Technologies and versions used (for reproducibility).
<br>
&nbsp;&nbsp;- <strong>Language:</strong> Python 3.12.10
<br>
&nbsp;&nbsp;- <strong>Core AI:</strong> PyTorch 2.9.1+cu126 (CUDA required)
<br>
&nbsp;&nbsp;- <strong>Audio:</strong> Librosa 0.11.0, PyDub 0.25.1
<br>
&nbsp;&nbsp;- <strong>GUI/Video:</strong> PyQt6 6.10.2, SpoutGL, OpenGL
<br>
&nbsp;&nbsp;- <strong>Models:</strong> StyleGAN3 (pretrained snapshot), OpenAI CLIP, Custom MLP.

### Results

- Results of the best configuration.
<br>
Tested on an <strong>NVIDIA RTX 4060</strong>, the system maintains a stable <strong>30 FPS</strong> at 256x256 resolution (upscalable in post-processing). The perceived latency is under 100ms, providing effective audio-visual synchronization. The Audio-to-Color mapping (e.g., Heavy Bass = Red/Fire) is consistent due to supervised CLIP training.

- Ablation Study: Comparison across configurations.
<br>
Without the <strong>MLP</strong>, connecting audio features directly to the latent space resulted in nonsensical, flickering images. Without <strong>RMS-based step modulation</strong>, transitions were either too slow during drops or too jittery during ambient sections.

## 4. Discussion and Conclusions

### Results Discussion

- Analysis of performance in relation to expectations.
<br>
The project met its performance expectations. Thanks to the <strong>multithreaded optimization</strong>, we achieved a stable application suitable for real-time usage.

### Method Validity

- Evaluation if the method meets expectations.
<br>
Using <strong>CLIP</strong> to "label" the latent space proved superior to random generation. It ensures the visual reactivity is <strong>semantic</strong>—changing the *atmosphere* of the video—rather than just mechanical movement.

### Limitations and Maturity

- Limits of applicability and biases.
<br>
&nbsp;&nbsp;* <strong>Hardware:</strong> Strictly requires an NVIDIA GPU with <strong>CUDA</strong> support.
<br>
&nbsp;&nbsp;* <strong>Fixed Domain:</strong> The model can only generate imagery within the domain of its training dataset.
<br>
&nbsp;&nbsp;* <strong>Resolution:</strong> Currently limited to 256x256px to maintain real-time framerates.

### Future Works

- Proposals to advance the project.
<br>
Future improvements could include <strong>TensorRT</strong> optimization to increase resolution or FPS. Additionally, implementing a predictive <strong>Beat Tracking</strong> algorithm would allow the system to anticipate musical drops, further tightening visual synchronization.