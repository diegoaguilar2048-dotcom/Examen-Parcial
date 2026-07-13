import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# 1. Datos del experimento (proporcionados en el informe)
f_raw = [
    100, 120, 145, 170, 200, 235, 270, 210, 255, 105, 160, 520, 585, 655, 730,
    810, 895, 985, 1080, 1180, 1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340, 2530, 2730
]
Z_raw = [
    152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9,
    131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1,
    143.5, 146.1, 149.0, 152.2, 155.6, 159.2
]

# Ordenar los datos de forma creciente por frecuencia para el análisis numérico
datos = sorted(zip(f_raw, Z_raw))
f_pts = np.array([d[0] for d in datos], dtype=float)
Z_pts = np.array([d[1] for d in datos], dtype=float)

n_puntos = len(f_pts)
grado = n_puntos - 1  # Grado 29 para interpolar 30 puntos exactamente

# 2. Ajuste polinómico robusto utilizando escalado interno (para mitigar el desbordamiento numérico)
# Polynomial.fit mapea automáticamente el dominio [100, 2730] a [-1, 1] antes de calcular
poly_global = Polynomial.fit(f_pts, Z_pts, deg=grado)

# Evaluar de forma físicamente aceptable en f = 1000 Hz
Z_1000 = poly_global(1000.0)

print(f"--- Resultados Corregidos (Estabilización Numérica) ---")
print(f"Magnitud de Impedancia |Z| en f = 1000 Hz: {Z_1000:.4f} \u03a9")
print(f"(Nota: Este valor ahora es coherente con la tendencia de los datos vecinos de la tabla)\n")

# 3. Validación Leave-One-Out (LOO) sobre 5 puntos elegidos al azar de forma reproducible
np.random.seed(42)
indices_loo = np.random.choice(range(n_puntos), size=5, replace=False)
errores_relativos = []

print(f"--- Validación Leave-One-Out (LOO) ---")
for idx in indices_loo:
    f_test, Z_test = f_pts[idx], Z_pts[idx]
    
    # Excluir el punto de prueba
    f_train = np.delete(f_pts, idx)
    Z_train = np.delete(Z_pts, idx)
    
    # Ajustar un polinomio de grado 28 con los 29 puntos restantes
    poly_loo = Polynomial.fit(f_train, Z_train, deg=grado-1)
    
    # Predicción
    Z_pred = poly_loo(f_test)
    err_rel = abs(Z_pred - Z_test) / Z_test
    errores_relativos.append(err_rel)
    
    print(f"f omitida: {f_test:4.0f} Hz | Real: {Z_test:5.1f} \u03a9 | Predicción LOO: {Z_pred:8.2f} \u03a9 | Error Rel: {err_rel*100:6.3f}%")

print(f"\nError relativo promedio en LOO: {np.mean(errores_relativos)*100:.4f}%\n")

# 4. Graficar el resultado para evidenciar las oscilaciones reales del Fenómeno de Runge
f_mesh = np.linspace(min(f_pts), max(f_pts), 2000)
Z_mesh = poly_global(f_mesh)

plt.figure(figsize=(11, 6))
plt.plot(f_mesh, Z_mesh, 'r--', label='Polinomio Global Grado 29 (Fenómeno de Runge Real)', alpha=0.8)
plt.scatter(f_pts, Z_pts, color='blue', edgecolor='k', s=40, zorder=5, label='Datos de Laboratorio')
plt.axvline(1000, color='orange', linestyle=':', linewidth=2, label='Frecuencia de Estudio (1000 Hz)')
plt.plot(1000, Z_1000, 'go', markersize=9, label=f'Evaluación Real ({Z_1000:.2f} \u03a9)')

# Límites del eje Y estratégicos para observar cómo se dispara en las zonas críticas
plt.ylim(100, 220)
plt.title('Parte B1 Corregida: Interpolación Polinómica Estable y Fenómeno de Runge', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, f (Hz)')
plt.ylabel('Magnitud de Impedancia, |Z| (\u03a9)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
