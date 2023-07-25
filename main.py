import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Parâmetros
fs = 1000  # Frequência de Amostragem
N = 500  # Número de pontos do sinal
n = np.arange(N) / fs

sinal = 50 * n * np.exp(-15 * n)
ruido = 0.04 * np.random.randn(len(sinal))
sinal_ruidoso = sinal + ruido

# Plotando os sinais
plt.figure(figsize=(10, 6))

plt.plot(n, sinal, label='Sinal Original')
plt.plot(n, ruido, label='Ruído', alpha=0.7)
plt.plot(n, sinal_ruidoso, label='Sinal Ruidoso', linewidth=0.7)

# Adicionando título e legendas
plt.title('Sinal, Ruído e Sinal Ruidoso')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)



# Plotando espectro do sinal ruidoso
plt.figure()
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
X = np.fft.fftshift(np.fft.fft(sinal_ruidoso))
plt.plot(f, np.abs(X), linewidth=1.5)
plt.grid(True)
plt.title('Espectro do sinal ruidoso')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.xlim([-fs / 2, fs / 2])
plt.tight_layout()
plt.xticks([-fs / 2, -250, 0, 250, fs / 2], ['500', '-250', '0', '250', '500'])
plt.show()