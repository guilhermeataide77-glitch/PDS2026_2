import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = '{:.2e}'.format


def dft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.exp(-2j * np.pi * k * n / N)
    return np.dot(W, x)


def idft(X):
    X = np.asarray(X, dtype=complex)
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))

    W_inv = np.exp(2j * np.pi * k * n / N) / N
    return np.dot(W_inv, X)


x_teste = np.array([1, 2, 3, 4], dtype=float)
X_teste = dft(x_teste)
x_recuperado = idft(X_teste)

print("--- Teste Básico ---")
print("Sinal original:    ", x_teste)
print("DFT:              ", X_teste)
print("Sinal recuperado: ", np.real_if_close(x_recuperado))
print(f"Erro máximo:      {np.max(np.abs(x_teste - x_recuperado)):.2e}\n")


rng = np.random.default_rng(7)
N = 8
x = rng.normal(size=N) + 1j * rng.normal(size=N)
y = rng.normal(size=N) + 1j * rng.normal(size=N)

a, b, m = 2.0, -0.5, 2
X = dft(x)
Y = dft(y)

erro_linearidade = np.max(np.abs(dft(a*x + b*y) - (a*X + b*Y)))

x_deslocado = np.roll(x, m)
X_teorico = X * np.exp(-2j * np.pi * np.arange(N) * m / N)
erro_deslocamento = np.max(np.abs(dft(x_deslocado) - X_teorico))

indices = (-np.arange(N)) % N
erro_conjugacao = np.max(np.abs(dft(np.conj(x)) - np.conj(X[indices])))

energia_tempo = np.sum(np.abs(x)**2)
energia_frequencia = np.sum(np.abs(X)**2) / N
erro_parseval = abs(energia_tempo - energia_frequencia)

erro_idft = np.max(np.abs(idft(X) - x))

X_numpy = np.fft.fft(x)
x_numpy_inv = np.fft.ifft(X)
erro_dft_numpy = np.max(np.abs(X - X_numpy))
erro_idft_numpy = np.max(np.abs(idft(X) - x_numpy_inv))

resultados = pd.DataFrame({
    "Propriedade / Validação": [
        "Linearidade",
        "Deslocamento circular",
        "Conjugação",
        "Parseval",
        "DFT seguida de IDFT",
        "DFT manual vs np.fft.fft",
        "IDFT manual vs np.fft.ifft"
    ],
    "Erro Máximo": [
        erro_linearidade,
        erro_deslocamento,
        erro_conjugacao,
        erro_parseval,
        erro_idft,
        erro_dft_numpy,
        erro_idft_numpy
    ]
})

print("--- Validação de Propriedades e Referência NumPy ---")
display(resultados)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.stem(np.arange(N), np.abs(x), basefmt="k-")
plt.title("Magnitude do Sinal no Tempo |x[n]|")
plt.xlabel("n")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(np.arange(N), np.abs(X), basefmt="k-")
plt.title("Espectro de Magnitude |X[k]|")
plt.xlabel("k")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
