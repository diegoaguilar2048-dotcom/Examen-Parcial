import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Datos experimentales originales (30 puntos)
f = np.array([100, 150, 200, 250, 320, 390, 440, 510, 580, 640, 730, 820, 910, 1000, 1070, 
              1150, 1240, 1330, 1420, 1510, 1600, 1720, 1840, 1930, 2050, 2190, 2310, 2420, 2560, 2730])

Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 
              131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2])

# 2. Construcción del Spline Cúbico Natural
# bc_type='natural' garantiza S''(x_0) = S''(x_n) = 0
cs = CubicSpline(f, Z, bc_type='natural')

# 3. Derivación Analítica del Trazador utilizando el método integrado .derivative()
# Esto extrae matemáticamente la derivada exacta de los polinomios de tercer grado segmentarios
cs_deriv1 = cs.derivative(nu=1) # Primera derivada d|Z|/df
cs_deriv2 = cs.derivative(nu=2) # Segunda derivada d^2|Z|/df^2

# 4. Localización Numérica Exacta del Mínimo
# El mínimo ocurre de forma rigurosa donde la primera derivada es igual a 0.
# Generamos una malla de alta densidad para localizar el cruce por cero de la derivada.
f_dense = np.linspace(f.min(), f.max(), 10000)
deriv1_dense = cs_deriv1(f_dense)

# Buscamos el índice donde cambia de signo (de negativo a positivo)
idx_minimo = np.where(np.diff(np.sign(deriv1_dense)) > 0)[0][0]
f_min_exacto = f_dense[idx_minimo]
Z_min_exacto = cs(f_min_exacto)
deriv2_min_exacta = cs_deriv2(f_min_exacto)

print("--- ANÁLISIS DE DERIVACIÓN Y MÍNIMO CRÍTICO ---")
print(f"Frecuencia exacta del mínimo local (f_min): {f_min_exacto:.4f} Hz")
print(f"Magnitud de impedancia en dicho punto (|Z|_min): {Z_min_exacto:.4f} Ohms")
print(f"Valor de la segunda derivada en el punto crítico: {deriv2_min_exacta:.6f} Ohms/Hz^2")

# 5. Configuración de la Ventana Gráfica Completa (Subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Gráfica Superior: Spline de Impedancia original
ax1.plot(f_dense, cs(f_dense), color='blue', linewidth=2, label='Spline Cúbico Natural $|Z|(f)$')
ax1.scatter(f, Z, color='black', s=25, label='Datos Experimentales', zorder=3)
ax1.scatter(f_min_exacto, Z_min_exacto, color='red', s=80, marker='*', label=f'Mínimo Exacto ({f_min_exacto:.1f} Hz)', zorder=5)
ax1.set_ylabel('Impedancia, $|Z|$ ($\Omega$)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right')
ax1.set_title('Derivación Analítica del Espectro de Impedancia Bioeléctrica', fontsize=12, fontweight='bold')

# Gráfica Inferior: Comportamiento de las Derivadas Analíticas
ax2.plot(f_dense, deriv1_dense, color='darkorange', linewidth=1.8, label="Primera Derivada $d|Z|/df$ (Pendiente)")
ax2.plot(f_dense, cs_deriv2(f_dense), color='purple', linewidth=1.2, linestyle=':', label="Segunda Derivada $d^2|Z|/df^2$ (Curvatura)")
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7) # Línea de referencia y=0
ax2.scatter(f_min_exacto, 0, color='red', s=50, zorder=5, label='Cruce por Cero ($d|Z|/df = 0$)')
ax2.set_xlabel('Frecuencia, $f$ (Hz)', fontsize=10)
ax2.set_ylabel('Razón de Cambio', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()
