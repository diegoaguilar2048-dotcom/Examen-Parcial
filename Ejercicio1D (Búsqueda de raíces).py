import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Datos experimentales originales (30 puntos)
f = np.array([100, 160, 220, 290, 360, 420, 490, 560, 620, 690, 750, 810, 880, 940, 1000, 
              1070, 1130, 1200, 1270, 1330, 1400, 1460, 1530, 1600, 1660, 1730, 1800, 1860, 1930, 2000,
              2060, 2130, 2200, 2260, 2330, 2400, 2460, 2530, 2600, 2660, 2730])[:30] # Limitado a 30 puntos según la guía

Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 
              131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2])

# 2. Reconstrucción del Spline Cúbico Natural y sus Derivadas Analíticas
cs = CubicSpline(f, Z, bc_type='natural')
cs_deriv1 = cs.derivative(nu=1)

# Definición de la función de umbral g(f) = |Z|(f) - 150
def g(freq):
    return cs(freq) - 150.0

def g_deriv(freq):
    return cs_deriv1(freq)

# 3. Implementación Manual del Método de Bisección
def biseccion(xl, xu, tol=1e-6, max_iter=100):
    historial = []
    if g(xl) * g(xu) >= 0:
        return None, historial
    for i in range(max_iter):
        xr = (xl + xu) / 2.0
        error = abs(xu - xl) / 2.0
        historial.append((i+1, xr, g(xr), error))
        if abs(g(xr)) < tol or error < tol:
            break
        if g(xl) * g(xr) < 0:
            xu = xr
        else:
            xl = xr
    return xr, historial

# 4. Implementación Manual del Método de Newton-Raphson
def newton_raphson(x0, tol=1e-6, max_iter=100):
    historial = []
    xr = x0
    for i in range(max_iter):
        f_val = g(xr)
        d_val = g_deriv(xr)
        if d_val == 0:
            break
        xr_next = xr - f_val / d_val
        error = abs(xr_next - xr)
        historial.append((i+1, xr, f_val, error))
        if error < tol or abs(f_val) < tol:
            xr = xr_next
            break
        xr = xr_next
    return xr, historial

# 5. Cálculo de Raíces para Delimitar la Banda Segura (|Z| = 150 Ohms)
# Raíz 1 (Límite Inferior): Buscada en el intervalo [100, 200] Hz
raiz1_bis, hist1_bis = biseccion(100, 200)
raiz1_nr, hist1_nr = newton_raphson(130)

# Raíz 2 (Límite Superior): Buscada en el intervalo [1800, 2000] Hz
raiz2_bis, hist2_bis = biseccion(1800, 2000)
raiz2_nr, hist2_nr = newton_raphson(1900)

print("=== TABLA DE CONVERGENCIA COMPARATIVA ===")
print(f"Raíz 1 (Límite Inferior): Bisección = {raiz1_bis:.4f} Hz (Iter: {len(hist1_bis)}) | Newton-Raphson = {raiz1_nr:.4f} Hz (Iter: {len(hist1_nr)})")
print(f"Raíz 2 (Límite Superior): Bisección = {raiz2_bis:.4f} Hz (Iter: {len(hist2_bis)}) | Newton-Raphson = {raiz2_nr:.4f} Hz (Iter: {len(hist2_nr)})\n")

# 6. Cálculo de la Sensibilidad Inversa en f ≈ 2000 Hz
f_sens = 2000.0
derivada_sens = cs_deriv1(f_sens)
sensibilidad_inversa = 1.0 / derivada_sens
print("=== ANÁLISIS DE SENSIBILIDAD INVERSA ===")
print(f"Primera derivada analítica en {f_sens} Hz: {derivada_sens:.6f} Ohms/Hz")
print(f"Sensibilidad Inversa (df/d|Z|) en {f_sens} Hz: {sensibilidad_inversa:.4f} Hz/Ohm\n")

# 7. Despliegue Gráfico de Confinamiento de Banda
f_dense = np.linspace(f.min(), f.max(), 1000)
plt.figure(figsize=(9, 5))
plt.plot(f_dense, cs(f_dense), color='blue', linewidth=2, label='Spline Cúbico Natural $|Z|(f)$')
plt.axhline(150, color='red', linestyle='--', alpha=0.8, label='Umbral Crítico ($150\ \Omega$)')

# Resaltar raíces y área segura
plt.scatter([raiz1_nr, raiz2_nr], [150, 150], color='black', s=60, zorder=5, label='Fronteras de Banda')
plt.fill_between(f_dense, cs(f_dense), 150, where=(f_dense >= raiz1_nr) & (f_dense <= raiz2_nr), 
                 color='green', alpha=0.15, label='Banda de Operación Segura')

plt.title('Delimitación de la Banda Segura mediante Búsqueda de Raíces', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, f (Hz)', fontsize=10)
plt.ylabel('Magnitud de Impedancia, |Z| (Ohms)', fontsize=10)
plt.xlim(f.min(), f.max())
plt.ylim(125, 165)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper center')
plt.tight_layout()
plt.show()
