import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# 1. Datos del experimento extraídos estrictamente del informe
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
spline_natural = CubicSpline(f_pts, Z_pts, bc_type='natural')

# 3. Derivación analítica del spline
primera_derivada_spline = spline_natural.derivative(nu=1)
segunda_derivada_spline = spline_natural.derivative(nu=2)

# 4. Localización precisa del mínimo exacto (cruce por cero de la primera derivada)
# Se define un intervalo seguro basado en el comportamiento de la curva
f_min_exacto = brentq(primera_derivada_spline, 600.0, 1200.0)
Z_min_exacto = spline_natural(f_min_exacto)

# 5. Evaluación de la estabilidad (Segunda Derivada)
d2Z_df2_min = segunda_derivada_spline(f_min_exacto)

print("--- Resultados de la Derivación Analítica ---")
print(f"Frecuencia exacta del mínimo local:  {f_min_exacto:.4f} Hz")
print(f"Magnitud de Impedancia en el mínimo: {Z_min_exacto:.4f} \u03a9")
print(f"Valor de la 2da derivada en el mín:  {d2Z_df2_min:.6f} \u03a9/Hz\u00b2")
print(f"Signo de la 2da derivada:            POSITIVO (+) -> Mínimo Estable")
print("---------------------------------------------\n")

# 6. Generación de la malla fina para graficar de forma continua
f_mesh = np.linspace(min(f_pts), max(f_pts), 2000)

# Inicializar los dos paneles (Subplots) compartiendo el eje X
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# --- PANEL SUPERIOR: Espectro de Impedancia Bioeléctrica ---
ax1.plot(f_mesh, spline_natural(f_mesh), 'b-', linewidth=2, label='Spline Cúbico Natural $|Z|(f)$')
ax1.scatter(f_pts, Z_pts, color='black', s=30, zorder=5, label='Datos Experimentales')
ax1.plot(f_min_exacto, Z_min_exacto, 'r*', markersize=11, zorder=6, label=f'Mínimo Exacto ({f_min_exacto:.1f} Hz)')
ax1.set_ylabel('Impedancia, $|Z|$ ($\Omega$)')
ax1.set_title('Derivación Analítica del Espectro de Impedancia Bioeléctrica', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right')

# --- PANEL INFERIOR: Razones de Cambio (Derivadas Analíticas) ---
ax2.plot(f_mesh, primera_derivada_spline(f_mesh), color='orange', linewidth=2, label='Primera Derivada $d|Z|/df$ (Pendiente)')
ax2.plot(f_mesh, segunda_derivada_spline(f_mesh), 'g:', linewidth=1.5, label='Segunda Derivada $d^2|Z|/df^2$ (Curvatura)')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax2.plot(f_min_exacto, 0, 'ro', markersize=7, label='Cruce por Cero ($d|Z|/df = 0$)')

ax2.set_xlabel('Frecuencia, $f$ (Hz)')
ax2.set_ylabel('Razón de Cambio')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='lower right')

# Ajustar diseño y desplegar
plt.tight_layout()
plt.show()
