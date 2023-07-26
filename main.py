

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Cria um sinal de teste com várias componentes de frequência
fs = 100  # Frequência de amostragem original
t = np.arange(0, 5, 1/fs)  # Agora o sinal tem duração de 10 segundos
sinal = np.sin(2*np.pi*5*t) + 0.8*np.sin(2*np.pi*15*t) + 0.5*np.sin(2*np.pi*30*t) + 0.2*np.sin(2*np.pi*40*t)
# ruido_branco =  0.3 * np.random.randn(len(sinal))
# f_portadora = 3500
# portadora = 0.4*np.cos(2 * np.pi * f_portadora * t)
# ruido = ruido_branco * portadora
# sinal += ruido

# Projeta um filtro equiripple para remover componentes acima de 8 Hz
N = 100  # Ordem do filtro
frequencias = [0, 8, 10, fs/2]
ganhos = [1, 0]
filtro = sp.signal.remez(N, frequencias, ganhos, fs=fs)

# Aplica o filtro ao sinal
sinal_filtrado = sp.signal.lfilter(filtro, 1, sinal)

# Resposta em frequência do filtro
freqs, response = sp.signal.freqz(filtro, worN=8000)
freqs = freqs * fs / (2*np.pi)  # Converte para Hz

ordens = [10, 30, 50, 70, 80, 100, 120, 200]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i in range(4):
  # Selecione o subplot atual
  ax = axs[i // 2, i % 2]

  # Loop através das duas ordens no par atual
  for ordem in ordens[i*2:i*2+2]:
    equiripple = sp.signal.remez(ordem, frequencias, ganhos, fs=fs)
    w, H = sp.signal.freqz(equiripple, [1], worN=8000, fs=fs)
    ax.plot(w, abs(H), linewidth=1.5, label=f"Ordem = {ordem}")

  # Configuração do subplot
  ax.set_title(
      f'Resposta em Frequência das Ordens {ordens[i*2]} e {ordens[i*2+1]}')
  ax.set_xlabel('Frequência [Hz]')
  ax.set_ylabel('Magnitude')
  ax.grid(True)
  ax.legend()

plt.subplots_adjust(hspace=0.3)
  

# Plota o sinal no domínio do tempo
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, sinal)
plt.title('Sinal Original')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.grid(True)

# Plota o sinal filtrado no domínio do tempo
plt.subplot(2, 1, 2)
plt.plot(t, sinal_filtrado)
plt.title('Sinal Filtrado')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplots_adjust(hspace=0.3)


# Diagrama polos e zeros
equiripple = sp.signal.remez(N, frequencias, ganhos, fs=fs)
zeros, poles, gain = sp.signal.tf2zpk(equiripple, [1])
plt.figure()
plt.scatter(np.real(zeros), np.imag(zeros),
            marker='o', color='b', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles),
            marker='x', color='r', label='Polos')
plt.title(
    f'Diagrama de Polos e Zeros (Ordem do filtro igual a {N})')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend()
plt.grid(True)
plt.axis('equal')



def plot_fft(sinal, fs, title, subplot):
    N = len(sinal)
    T = 1.0 / fs
    yf = np.fft.fft(sinal)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    subplot.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    subplot.set_title(title)
    subplot.set_xlabel('Frequência [Hz]')
    subplot.set_ylabel('Magnitude')
    subplot.grid(True)

plt.figure(figsize=(10, 8))

# Subplot 1: FFT do Sinal Original
ax1 = plt.subplot(2, 1, 1)
plot_fft(sinal, fs, 'FFT do Sinal Original', ax1)

# Subplot 2: FFT do Sinal Filtrado
ax2 = plt.subplot(2, 1, 2)
plot_fft(sinal_filtrado, fs, 'FFT do Sinal Filtrado', ax2)

plt.tight_layout()
plt.show()