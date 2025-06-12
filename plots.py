import matplotlib.pyplot as plt
import numpy as np

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle"
)

# Dati dalla tabella
tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]
tempi_jacobi = [0.0513, 0.0883, 0.1391, 0.1766]
tempi_gauss_seidel = [2.3373, 4.9140, 6.2016, 9.4444]
tempi_gradiente = [0.0687, 0.1213, 0.1719, 0.2374]
tempi_grad_coniugato = [0.0012, 0.0017, 0.0022, 0.0025]


# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(tolleranze, tempi_jacobi, marker="o", label="Jacobi")
plt.plot(tolleranze, tempi_gauss_seidel, marker="s", label="Gauss-Seidel")
plt.plot(tolleranze, tempi_gradiente, marker="^", label="Gradiente")
plt.plot(tolleranze, tempi_grad_coniugato, marker="d", label="Gradiente Coniugato")

plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Tolleranza")
plt.ylabel("Tempo di esecuzione (s)")
plt.title("Tempo di esecuzione vs Tolleranza")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("./plots/confronto_tempi_vem1.png", dpi=300)


# # Iterazioni dalla tabella
# iter_jacobi = [1315, 2434, 3553, 4672]
# iter_gauss_seidel = [660, 1219, 1779, 2339]
# iter_gradiente = [891, 1613, 2337, 3059]
# iter_grad_coniugato = [38, 45, 53, 59]


# # Creazione del grafico
# plt.figure(figsize=(10, 6))
# plt.plot(tolleranze, iter_jacobi, marker="o", label="Jacobi")
# plt.plot(tolleranze, iter_gauss_seidel, marker="s", label="Gauss-Seidel")
# plt.plot(tolleranze, iter_gradiente, marker="^", label="Gradiente")
# plt.plot(tolleranze, iter_grad_coniugato, marker="d", label="Gradiente Coniugato")

# plt.xscale("log")
# plt.xlabel("Tolleranza")
# plt.ylabel("Numero di Iterazioni")
# plt.title("Numero di Iterazioni vs Tolleranza")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# plt.tight_layout()

# plt.savefig("./plots/iterazioni_vem1.png", dpi=300)
