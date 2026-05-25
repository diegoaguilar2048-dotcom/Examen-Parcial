import numpy as np
import matplotlib.pyplot as plt

# 1. Vectores de datos experimentales originales (30 puntos)
f = np.array([100, 140, 180, 220, 270, 320, 380, 440, 500, 570, 650, 730, 820, 920, 1030, 
              1150, 1280, 1420, 1570, 1730, 1900, 2030, 2150, 2260, 2360, 2450, 2530, 2600, 2670, 2730])

Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 
              131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2])

# 2. Polinomio Global Completo (Grado 29) con normalización para evitar desbordamiento
f_mean, f_std = f.mean(), f.std()
f_scaled = (f - f_mean) / f_std
p_global = np.polyfit(f_scaled, Z, 29)

# Interpolación en f = 1000 Hz
Z_1000_pred = np.polyval(p_global, (1000 - f_mean) / f_std)
print(f"Valor interpolado en f = 1000 Hz: {Z_1000_pred:.4f} Ohms")

# 3. Validación Cruzada (LOO 5 puntos) con escala controlada por separado
np.random.seed(42) # Semilla fija para consistencia en la selección aleatoria
indices_test = np.random.choice(len(f), size=5, replace=False)

f_train = np.delete(f, indices_test)
Z_train = np.delete(Z, indices_test)

f_tr_mean, f_tr_std = f_train.mean(), f_train.std()
f_train_scaled = (f_train - f_tr_mean) / f_tr_std
p_train = np.polyfit(f_train_scaled, Z_train, len(f_train) - 1)

f_test_scaled = (f[indices_test] - f_tr_mean) / f_tr_std
Z_test_pred = np.polyval(p_train, f_test_scaled)
Z_test_real = Z[indices_test]

errores_relativos = np.abs((Z_test_real - Z_test_pred) / Z_test_real)
error_medio_loo = np.mean(errores_relativos) * 100
print(f"Error relativo medio estimado (LOO 5 puntos): {error_medio_loo:.4f}%")

# 4. Bloque de Graficación (Genera la curva continua y los puntos)
f_fine = np.linspace(f.min(), f.max(), 1000)
f_fine_scaled = (f_fine - f_mean) / f_std
Z_fine = np.polyval(p_global, f_fine_scaled)

plt.figure(figsize=(9, 5))
plt.plot(f_fine, Z_fine, color='red', linewidth=1.5, label='Polinomio Global (Grado 29)')
plt.scatter(f, Z, color='blue', s=25, label='Datos Experimentales', zorder=3)
plt.scatter(f[indices_test], Z[indices_test], color='orange', s=60, edgecolor='black', label='Puntos omitidos en LOO', zorder=4)
plt.axvline(1000, color='green', linestyle='--', alpha=0.7, label='Frecuencia de Evaluación (1000 Hz)')

# Ajustamos límites del eje Y para observar las oscilaciones de Runge en los extremos
plt.ylim(50, 250) 
plt.title('Evidencia de Inestabilidad Numérica: Fenómeno de Runge', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia, f (Hz)', fontsize=10)
plt.ylabel('Magnitud de Impedancia, |Z| (Ohms)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper center')

# Forzar el despliegue de la ventana de la gráfica
plt.tight_layout()
plt.show()
