import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Parâmetros
fs = 20000  # Frequência de Amostragem
fc = 2000    # Frequência de corte
N = 10000  # Número de pontos do sinal
n = np.arange(N) / fs

sinal = 50 * n * np.exp(-15 * n)
ruido_branco = 0.04 * np.random.randn(len(sinal))
# Frequência da portadora para a modulação
f_portadora = 3500  # 2.5kHz
portadora = np.cos(2 * np.pi * f_portadora * n)
ruido = ruido_branco * portadora

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
plt.xticks([-fs / 2, -fs / 4, 0, fs/4, fs / 2], [f'{-fs / 2}', f'{-fs / 4}', '0', f'{fs / 2}', f'{-fs / 2}'])




# Características do filtro
# Tipos de filtro:
# passa-baixa -> tipo == 1, passa-faixa -> tipo == 2, passa-alta -> tipo == 3

banda_transicao = 100   # Banda de transição do filtro

tipo_filtro = 1

if tipo_filtro == 1:
  banda_passagem = [0, fc]
  frequencias = [banda_passagem[0], banda_passagem[1],
                 banda_passagem[1] + banda_transicao, fs / 2]
  magnitude = [1, 0]

elif tipo_filtro == 2:
  banda_passagem = [6000, 8000]
  frequencias = [0, (banda_passagem[0] - banda_transicao), banda_passagem[0],
                 banda_passagem[1], (banda_passagem[1] + banda_transicao), fs / 2]
  magnitude = [0, 1, 0]
elif tipo_filtro == 3:
  banda_passagem = [fc, fs / 2]
  frequencias = [0, banda_passagem[0] -
                 banda_transicao, banda_passagem[0], fs / 2]
  magnitude = [0, 1]
  
ordem_filtro = 100

equiripple = sp.signal.remez(ordem_filtro, frequencias, magnitude, fs=fs)
# Resposta em frequência do filtro
# Resposta em frequência do filtro
w, H = sp.signal.freqz(equiripple, [1], worN=fs//2, fs=fs)


# Plotando a resposta em frequência do filtro
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(H), linewidth=1.5, label=f"Ordem = {ordem_filtro}")
plt.title(f'Resposta em Frequência do Filtro Equiripple de Ordem {ordem_filtro}')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude ')
plt.grid()
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(w, 20 * np.log10(np.abs(H)), linewidth=1.5, label=f"Ordem = {ordem_filtro}")
plt.title(f'Resposta em Frequência do Filtro Equiripple de Ordem {ordem_filtro}')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.tight_layout()

# Aplicando o filtro ao sinal ruidoso
sinal_filtrado = sp.signal.lfilter(equiripple, [1], sinal_ruidoso)

# Plotando o sinal filtrado
plt.figure(figsize=(10, 6))
plt.plot(n, sinal_ruidoso, label='Sinal Ruidoso', linewidth=0.7)
plt.plot(n, sinal_filtrado, label='Sinal Filtrado', linewidth=1)
plt.title('Sinal Filtrado')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


# Plotando o espectro do sinal filtrado
X_filtered = np.fft.fftshift(np.fft.fft(sinal_filtrado))
plt.figure()
plt.plot(f, np.abs(X_filtered), linewidth=1.5)
plt.title('Espectro do sinal filtrado')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.xlim([-fs / 2, fs / 2])
plt.tight_layout()
plt.xticks([-fs / 2, -fs / 4, 0, fs/4, fs / 2], ['10000', '-5000', '0', '5000', '10000'])
plt.grid(True)
plt.show()
