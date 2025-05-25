import matplotlib.pyplot as plt
import numpy as np

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle"
)

# Dati dalla tabella
tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]
# tempi_jacobi = [0.0445, 0.0705, 0.0912, 0.1202]
# tempi_gauss_seidel = [0.2513, 0.4924, 0.6549, 1.0691]
# tempi_gradiente = [0.0492, 1.2519, 2.7377, 4.3979]
# tempi_grad_coniugato = [0.0095, 0.0265, 0.0340, 0.0372]

# # Creazione del grafico
# plt.figure(figsize=(10, 6))
# plt.plot(tolleranze, tempi_jacobi, marker="o", label="Jacobi")
# plt.plot(tolleranze, tempi_gauss_seidel, marker="s", label="Gauss-Seidel")
# plt.plot(tolleranze, tempi_gradiente, marker="^", label="Gradiente")
# plt.plot(tolleranze, tempi_grad_coniugato, marker="d", label="Gradiente Coniugato")

# plt.xscale("log")
# plt.xlabel("Tolleranza")
# plt.ylabel("Tempo di esecuzione (s)")
# plt.title("Tempo di esecuzione vs Tolleranza")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# plt.tight_layout()

# plt.savefig("./plots/confronto_tempi_spa1.png", dpi=300)


# # Iterazioni dalla tabella
# iter_jacobi = [108, 281, 406, 574]
# iter_gauss_seidel = [77, 171, 225, 352]
# iter_gradiente = [37, 582, 1136, 1772]
# iter_grad_coniugato = [14, 25, 29, 30]

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

# plt.savefig('./plots/iterazioni_spa1.png', dpi=300)

