import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
models = [
    'Weak-Supervision', 'Falcon-7B (0-Shot)', 'Llama-2-70B (0-Shot)', 
    'Falcon-7B (6-Shot)', 'Llama-2-70B (6-Shot)',
    'Bi-LSTM', 'BERT-base', 'FinBERT-base', 'FLANG-BERT-base', 'RoBERTa-base', 'RoBERTa-large'
]

latency_s = [
    0.000139, 7.2, 5.4,  # Weak-Supervision (s), Falcon-7B-Instruct (Zero-Shot), Llama-2-70B-Chat (Zero-Shot)
    9.6, 7.2,  # Falcon-7B-Instruct (Six-Shot), Llama-2-70B-Chat (Six-Shot)
    0.0012, 0.00246, 0.00246, 0.00246, 0.00240, 0.00810  # Fine-Tuned Models (s)
]

co2 = [
    2.42e-7, 0.118, 0.673,  # Weak-Supervision (g), Falcon-7B-Instruct (Zero-Shot), Llama-2-70B-Chat (Zero-Shot)
    0.157, 0.898,  # Falcon-7B-Instruct (Six-Shot), Llama-2-70B-Chat (Six-Shot)
    1.97e-5, 4.03e-5, 4.03e-5, 4.03e-5, 3.93e-5, 1.33e-4  # Fine-Tuned Models (g)
]

# Latency Plot with Log Scale (Horizontal Bar)
plt.figure(figsize=(8, 6))
plt.barh(models, latency_s, color='skyblue', hatch='//')
plt.xscale('log')
plt.xlabel('Latency (s) [Log Scale]')
plt.ylabel('Models')
plt.title('Latency Across Different Models (Log Scale)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()

# CO2 Emission Plot with Log Scale (Horizontal Bar)
plt.figure(figsize=(8, 6))
plt.barh(models, co2, color='salmon', hatch='\\\\')
plt.xscale('log')
plt.xlabel('CO2 Emissions (g) [Log Scale]')
plt.ylabel('Models')
plt.title('CO2 Emissions Across Different Models (Log Scale)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()
