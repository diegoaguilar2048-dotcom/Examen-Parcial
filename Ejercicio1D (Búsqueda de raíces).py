import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Datos del experimento extraídos del informe
f_raw = [
    100, 120, 145, 170, 200, 235, 270, 210, 255, 105, 160, 520, 585, 655, 730,
    810, 895, 985, 1080, 1180, 1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340, 2530, 2730
]
Z_raw = [
    152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9,
    131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1,
    143.5, 146.1, 149.0, 152.2, 155.6, 159.2
]

# Ordenar los datos cronológicamente por frecuencia
datos = sorted(zip(f_raw, Z_raw))
f_pts = np.array([d[0] for d in datos], dtype=float)
Z_pts = np.array([d[1] for d in datos], dtype=float)

# 2. Reconstrucción del modelo base (Spline Cúbico Natural)
spline = CubicSpline(f_pts, Z_pts, bc_type='natural')
derivada_spline = spline.derivative(nu=1)

# Definición de la función objetivo: g(f) = Z(f) - Z_th = 0
Z_th = 150.0
def g(f):
    return spline(f) - Z_th

def dg_df(f):
    return derivada_spline(f)

# 3. Implementación nativa del Método de Bisección
def biseccion(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("El intervalo no encierra una raíz (signos iguales).")
    iteracion = 0
    while (b - a) / 2.0 > tol and iteracion < max_iter:
        c = (a + b) / 2.0
        if func(c) == 0:
            return c, iteracion
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c
        iteracion += 1
    return (a + b) / 2.0, iteracion

# 4. Implementación nativa del Método de Newton-Raphson
def newton_raphson(func, dfunc, x0, tol=1e-6, max_iter=100):
    x = x0
    for iteracion in range(1, max_iter + 1):
        df_val = dfunc(x)
        if df_val == 0:
            raise ZeroDivisionError("Derivada igual a cero. El método se detiene.")
        x_nuevo = x - func(x) / df_val
        if abs(x_nuevo - x) < tol:
            return x_nuevo, iteracion
        x = x_nuevo
    return x, max_iter

# 5. Cálculo de las raíces exactas en los intervalos del problema
# Raíz 1 (Límite inferior): se encuentra entre 100 Hz y 200 Hz
r1_bis, iter_r1_bis = biseccion(g, 100.0, 200.0)
r1_nw, iter_r1_nw = newton_raphson(g, dg_df, x0=120.0)

# Raíz 2 (Límite superior): se encuentra entre 2100 Hz y 2500 Hz (atendiendo a la tabla real)
r2_bis, iter_r2_bis = biseccion(g, 2100.0, 2500.0)
r2_nw, iter_r2_nw = newton_raphson(g, dg_df, x0=2300.0)

print("=== COMPARATIVA DE MÉTODOS DE BÚSQUEDA DE RAÍCES ===")
print(f"LÍMITE INFERIOR (Raíz 1):")
print(f"  -> Bisección:      {r1_bis:.4f} Hz ({iter_r1_bis} iteraciones)")
print(f"  -> Newton-Raphson: {r1_nw:.4f} Hz ({iter_r1_nw} iteraciones)")
print(f"\nLÍMITE SUPERIOR (Raíz 2):")
print(f"  -> Bisección:      {r2_bis:.4f} Hz ({iter_r2_bis} iteraciones)")
print(f"  -> Newton-Raphson: {r2_nw:.4f} Hz ({iter_r2_nw} iteraciones)")
print("====================================================\n")

# 6. Análisis de Sensibilidad Inversa en la raíz más cercana a 2000 Hz (Raíz 2)
derivada_en_r2 = dg_df(r2_nw)         # dZ/df
sensibilidad_inversa = 1.0 / derivada_en_r2  # df/dZ

print("=== ANÁLISIS DE SENSIBILIDAD INVERSA (Cerca a 2000 Hz) ===")
print(f"Raíz analizada:                    {r2_nw:.4f} Hz")
print(f"Primera derivada d|Z|/df en raíz:  {derivada_en_r2:.6f} \u03a9/Hz")
print(f"Sensibilidad inversa df/d|Z|:       {sensibilidad_inversa:.4f} Hz/\u03a9")
print("==========================================================\n")

# 7. Generación del Gráfico Corregido de Delimitación de Banda Segura
f_mesh = np.linspace(100, 2500, 2000)
Z_mesh = spline(f_mesh)

plt.figure(figsize=(11, 6))
plt.plot(f_mesh, Z_mesh, 'b-', linewidth=2, label='Spline Cúbico Natural $|Z|(f)$')
plt.axhline(Z_th, color='r', linestyle='--', linewidth=1.5, label='Umbral Crítico (150 $\Omega$)')

# Remarcar las dos fronteras reales
plt.plot([r1_nw, r2_nw], [Z_th, Z_th], 'ko', markersize=8, label='Fronteras de Banda Calculadas')

# Colorear la verdadera Banda de Operación Segura (donde Z(f) < 150)
f_segura = np.linspace(r1_nw, r2_nw, 1000)
plt.fill_between(f_segura, spline(f_segura), Z_th, color='green', alpha=0.15, label='Banda de Operación Segura Real')

# Anotaciones de las raíces
plt.text(r1_nw + 20, Z_th + 1.5, f"$f_1 = {r1_nw:.2f}$ Hz", fontsize=10, fontweight='bold', color='black')
plt.text(r2_nw - 300, Z_th + 1.5, f"$f_2 = {r2_nw:.2f}$ Hz", fontsize=10, fontweight='bold', color='black')

plt.xlim(100, 2500)
plt.ylim(125, 160)
plt.title('Delimitación de la Banda Segura mediante Búsqueda de Raíces (Corregido)', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, $f$ (Hz)')
plt.ylabel('Magnitud de Impedancia, $|Z|$ ($\Omega$)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
