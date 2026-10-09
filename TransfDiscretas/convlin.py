
import numpy as np
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


def convolucao_direta(x, h):
    x = np.asarray(x)
    h = np.asarray(h)
    y = np.zeros(len(x) + len(h) - 1, dtype=complex)

    for n in range(len(y)):
        for k in range(len(x)):
            j = n - k
            if 0 <= j < len(h):
                y[n] += x[k] * h[j]

    return np.real_if_close(y)


def convolucao_dft(x, h):
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)

    N = len(x) + len(h) - 1

    x_pad = np.pad(x, (0, N - len(x)))
    h_pad = np.pad(h, (0, N - len(h)))

    X = dft(x_pad)
    H = dft(h_pad)

    Y = X * H
    y = idft(Y)

    return np.real_if_close(y)


x = np.array([1, 2, 3, 2], dtype=float)
h = np.array([1, -1, 2], dtype=float)

y_direta = convolucao_direta(x, h)
y_dft_manual = convolucao_dft(x, h)

y_numpy_convolve = np.convolve(x, h)

N = len(x) + len(h) - 1
x_pad = np.pad(x, (0, N - len(x)))
h_pad = np.pad(h, (0, N - len(h)))
y_numpy_fft = np.real_if_close(np.fft.ifft(np.fft.fft(x_pad) * np.fft.fft(h_pad)))

erro_vs_direta = np.max(np.abs(y_dft_manual - y_direta))
erro_vs_np_convolve = np.max(np.abs(y_dft_manual - y_numpy_convolve))
erro_vs_np_fft = np.max(np.abs(y_dft_manual - y_numpy_fft))

print("Sinal x[n]:            ", x)
print("Filtro h[n]:           ", h)
print("Convolução Manual:     ", y_dft_manual)
print("np.convolve (Oficial): ", y_numpy_convolve)
print("\n")

resultados = pd.DataFrame({
    "Comparação de Métodos": [
        "DFT Manual vs Convolução Direta",
        "DFT Manual vs np.convolve (Oficial)",
        "DFT Manual vs NumPy FFT/IFFT"
    ],
    "Erro Máximo": [
        erro_vs_direta,
        erro_vs_np_convolve,
        erro_vs_np_fft
    ]
})

display(resultados)
