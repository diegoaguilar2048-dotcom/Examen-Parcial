import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Datos experimentales originales (30 puntos)
f = np.array([100, 190, 280, 370, 460, 550, 640, 730, 820, 910, 1000, 1090, 1180, 1270, 1360, 
              1450, 1540, 1630, 1720, 1810, 1900, 1990, 2080, 2170, 2260, 2350, 2440, 2530, 2620, 2730])

Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 
              131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2])

# 2. RESOLUCIÓN DE LA PARTE B2: Spline Cúbico Natural
# bc_type='natural' impone de forma exacta que la segunda derivada en las fronteras exteriores sea cero.
cs_natural = CubicSpline(f, Z, bc_type='natural')

# Calcular el valor exacto interpolado por el Spline a f = 1000 Hz
Z_1000_spline = cs_natural(1000)
print(f"--- MÉTODOS NUMÉRICOS COMPULSAR ---")
print(f"Valor interpolado por Spline Cúbico en f = 1000 Hz: {Z_1000_spline:.4f} Ohms")

# 3. Datos de control del problema anterior (Polinomio de grado 29 estabilizado)
f_mean, f_std = f.mean(), f.std()
f_scaled = (f - f_mean) / f_std
p_global = np.polyfit(f_scaled, Z, 29)
Z_1000_poly = np.polyval(p_global, (1000 - f_mean) / f_std)
print(f"Valor interpolado por Polinomio Grado 29 en f = 1000 Hz: {Z_1000_poly:.4f} Ohms")

# 4. Generación de la Malla Fina solicitada en la guía
f_fine = np.linspace(f.min(), f.max(), 1000)
Z_spline_fine = cs_natural(f_fine)
Z_poly_fine = np.polyval(p_global, (f_fine - f_mean) / f_std)

# 5. Despliegue Gráfico Comparativo
plt.figure(figsize=(9, 5))
plt.plot(f_fine, Z_spline_fine, color='green', linewidth=2.0, label='Spline Cúbico Natural (Suave/Estable)')
plt.plot(f_fine, Z_poly_fine, color='red', linestyle='--', linewidth=1.2, label='Polinomio Global Grado 29 (Runge)')
plt.scatter(f, Z, color='blue', s=30, label='Datos de Laboratorio', zorder=5)
plt.axvline(1000, color='darkorange', linestyle=':', linewidth=1.5, label='Frecuencia bajo estudio (1000 Hz)')

# Ajustamos límites de visibilidad para notar el acople exacto
plt.ylim(120, 175)
plt.title('Validación de Modelos: Espectro de Impedancia Bioeléctrica', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, f (Hz)', fontsize=10)
plt.ylabel('Magnitud de Impedancia, |Z| (Ohms)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
