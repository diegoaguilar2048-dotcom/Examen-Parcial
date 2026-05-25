import matplotlib.pyplot as plt

# 1. Vectores de datos experimentales (30 pares ordenados)
frecuencia = [100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 585, 655, 730, 
              810, 895, 985, 1080, 1180, 1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340, 2530, 2730]

impedancia = [152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 
              131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2]

# 2. Creación del gráfico de dispersión (Scatter plot)
plt.figure(figsize=(8, 5))
plt.scatter(frecuencia, impedancia, color='blue', edgecolor='black', alpha=0.7, label='Datos Experimentales')

# 3. Resaltar el punto mínimo experimental
plt.scatter(730, 130.9, color='red', s=100, zorder=5, label='Mínimo Muestreado (730 Hz, 130.9 $\Omega$)')

# 4. Formato estético profesional
plt.title('Análisis Exploratorio: Magnitud de Impedancia vs. Frecuencia ($T = 37^\\circ$C)', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, $f$ (Hz)', fontsize=10)
plt.ylabel('Magnitud de Impedancia, $|Z|$ ($\\Omega$)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper center')
