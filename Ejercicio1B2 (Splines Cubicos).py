import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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

# Ordenar los datos de forma creciente por frecuencia
datos = sorted(zip(f_raw, Z_raw))
f_pts = np.array([d[0] for d in datos], dtype=float)
Z_pts = np.array([d[1] for d in datos], dtype=float)

# 2. Construcción del Spline Cúbico Natural
# bc_type='natural' impone que la segunda derivada en los extremos sea cero: S''(x0) = S''(xn) = 0
spline_natural = CubicSpline(f_pts, Z_pts, bc_type='natural')

# 3. Reajuste del Polinomio Global de Grado 29 (para la comparación visual)
poly_global = Polynomial.fit(f_pts, Z_pts, deg=len(f_pts)-1)

# 4. Evaluaciones en la Frecuencia Bajo Estudio (f = 1000 Hz)
Z_spline_1000 = spline_natural(1000.0)
Z_poly_1000 = poly_global(1000.0)

print(f"--- Comparativa de Interpolación en f = 1000 Hz ---")
print(f"Spline Cúbico Natural:       {Z_spline_1000:.4f} \u03a9")
print(f"Polinomio Global (Grado 29): {Z_poly_1000:.4f} \u03a9")
print(f"Diferencia absoluta:         {abs(Z_spline_1000 - Z_poly_1000):.6f} \u03a9\n")

# 5. Generación de la Malla Fina y Gráfico Comparativo
f_mesh = np.linspace(min(f_pts), max(f_pts), 2000)
Z_mesh_spline = spline_natural(f_mesh)
Z_mesh_poly = poly_global(f_mesh)

plt.figure(figsize=(11, 6))

# Trazado de las curvas de interpolación
plt.plot(f_mesh, Z_mesh_spline, 'g-', linewidth=2.5, label='Spline Cúbico Natural (Suave/Estable)')
plt.plot(f_mesh, Z_mesh_poly, 'r--', linewidth=1.2, label='Polinomio Global Grado 29 (Runge)')

# Puntos de datos de laboratorio
plt.scatter(f_pts, Z_pts, color='blue', edgecolor='b', s=35, zorder=5, label='Datos de Laboratorio')

# Marcadores para la frecuencia de estudio (1000 Hz)
plt.axvline(1000, color='orange', linestyle=':', linewidth=1.8, label='Frecuencia bajo estudio (1000 Hz)')
plt.plot(1000, Z_spline_1000, 'mo', markersize=8, zorder=6, label=f'Evaluación Spline ({Z_spline_1000:.4f} \u03a9)')

# Configuración estética idéntica a los requerimientos del informe
plt.ylim(120, 175)
plt.title('Validación de Modelos: Espectro de Impedancia Bioeléctrica', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, f (Hz)')
plt.ylabel('Magnitud de Impedancia, |Z| (Ohms)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()

plt.show()
