

import numpy as np
import matplotlib.pyplot as plt


def convolucao_circular_dft(x, h, N):
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)

    if len(x) > N or len(h) > N:
        raise ValueError("Os sinais não podem ser maiores que N.")

    x_pad = np.pad(x, (0, N - len(x)))
    h_pad = np.pad(h, (0, N - len(h)))

    return idft(dft(x_pad) * dft(h_pad))


def overlap_add(x, h, N):
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)

    M = len(h)

    if N < M:
        raise ValueError("N deve ser maior ou igual a M.")

    L = N - M + 1

    saida = np.zeros(len(x) + M - 1, dtype=complex)

    for inicio in range(0, len(x), L):
        bloco = x[inicio:inicio + L]

        y_bloco = convolucao_circular_dft(bloco, h, N)

        comprimento = len(bloco) + M - 1

        saida[inicio:inicio + comprimento] += (
            y_bloco[:comprimento]
        )

    return np.real_if_close(saida)


def overlap_save(x, h, N):
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)

    M = len(h)

    if N < M:
        raise ValueError("N deve ser maior ou igual a M.")

    L = N - M + 1
    tamanho_saida = len(x) + M - 1

    x_ext = np.concatenate([
        np.zeros(M - 1, dtype=complex),
        x
    ])

    numero_blocos = int(np.ceil(tamanho_saida / L))
    tamanho_necessario = (numero_blocos - 1) * L + N

    if len(x_ext) < tamanho_necessario:
        x_ext = np.pad(
            x_ext,
            (0, tamanho_necessario - len(x_ext))
        )

    saida = []

    for inicio in range(0, numero_blocos * L, L):
        bloco = x_ext[inicio:inicio + N]

        y_circular = convolucao_circular_dft(
            bloco, h, N
        )

        amostras_validas = y_circular[M - 1:]

        saida.extend(amostras_validas.tolist())

    return np.real_if_close(np.asarray(saida[:tamanho_saida]))


rng = np.random.default_rng(10)

x = rng.normal(size=20)
h = np.array([0.25, 0.5, 0.25, -0.1])

N = 8

y_referencia = convolucao_direta(x, h)
y_oa = overlap_add(x, h, N)
y_os = overlap_save(x, h, N)

print("Comprimento esperado:", len(x) + len(h) - 1)
print("Erro overlap-add:",
      np.max(np.abs(y_referencia - y_oa)))
print("Erro overlap-save:",
      np.max(np.abs(y_referencia - y_os)))

print("Overlap-add correto:",
      np.allclose(y_referencia, y_oa, atol=1e-10))
print("Overlap-save correto:",
      np.allclose(y_referencia, y_os, atol=1e-10))

plt.figure(figsize=(10, 4))
plt.plot(y_referencia, "o-", label="Convolução direta")
plt.plot(y_oa, "x--", label="Overlap-add")
plt.plot(y_os, ".:", label="Overlap-save")
plt.xlabel("Amostra n")
plt.ylabel("Amplitude")
plt.title("Comparação dos métodos de convolução")
plt.grid(True)
plt.legend()
plt.show()
