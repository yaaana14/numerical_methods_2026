import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. Визначення функції навантаження
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# Параметри інтегрування
a, b = 0, 24

# 2. Знаходження точного значення інтегралу
I0, _ = quad(f, a, b)


# 3. Функція для обчислення інтегралу методом Сімпсона
def simpson_method(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)

    I = y[0] + y[-1]
    I += 4 * np.sum(y[1:-1:2])
    I += 2 * np.sum(y[2:-2:2])
    return (h / 3) * I


# 4. Дослідження залежності точності від N
N_range = np.arange(10, 1001, 2)
errors = [abs(simpson_method(f, a, b, n) - I0) for n in N_range]

# Пошук N_opt для точності 1e-12
N_opt = 0
for n in range(10, 5000, 2):
    if abs(simpson_method(f, a, b, n) - I0) < 1e-12:
        N_opt = n
        break

# 5. Обчислення похибки при N0 (кратне 8)
N0 = int(N_opt / 10)
if N0 < 8: N0 = 8
N0 = N0 + (8 - N0 % 8) if N0 % 8 != 0 else N0
I_N0 = simpson_method(f, a, b, N0)
eps0 = abs(I_N0 - I0)

# 6. Метод Рунге-Ромберга
I_N0_half = simpson_method(f, a, b, N0 // 2)
I_Runge = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_Runge - I0)

# 7. Метод Ейткена
q = 2
I1 = simpson_method(f, a, b, N0)
I2 = simpson_method(f, a, b, N0 * q)
I3 = simpson_method(f, a, b, N0 * q ** 2)
I_Aitken = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))
p_aitken = np.log(abs((I3 - I2) / (I2 - I1))) / np.log(q)


# 9. Адаптивний алгоритм
def adaptive_simpson(f, a, b, eps, whole):
    mid = (a + b) / 2
    left = simpson_method(f, a, mid, 2)
    right = simpson_method(f, mid, b, 2)
    if abs(left + right - whole) <= 15 * eps:
        return left + right + (left + right - whole) / 15
    return adaptive_simpson(f, a, mid, eps / 2, left) + \
        adaptive_simpson(f, mid, b, eps / 2, right)


I_adaptive = adaptive_simpson(f, a, b, 1e-12, simpson_method(f, a, b, 2))

# --- ВИВІД РЕЗУЛЬТАТІВ ---
print(f"Точне значення I0: {I0:.12f}")
print(f"Оптимальне N_opt: {N_opt}")
print(f"Похибка при N0={N0}: {eps0:.2e}")
print(f"Похибка за Рунге-Ромбергом: {epsR:.2e}")
print(f"Похибка за Ейткеном: {abs(I_Aitken - I0):.2e}")
print(f"Порядок точності (Ейткен): {p_aitken:.2f}")

# --- ПОБУДОВА ГРАФІКІВ ---
plt.figure(figsize=(14, 6))

# Перший графік: Функція навантаження
plt.subplot(1, 2, 1)
x_plot = np.linspace(a, b, 1000)
plt.plot(x_plot, f(x_plot), label='Навантаження f(x)')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()

# Другий графік
plt.subplot(1, 2, 2)
plt.semilogy(N_range[:100], errors[:100], color='tab:red', marker='.', markersize=4, label='Похибка Simpson')
plt.title('Деталізована залежність похибки від N')
plt.xlabel('Кількість розбиттів N')
plt.ylabel('Похибка (логарифмічна шкала)')
plt.ylim(1e-13, 1e-1)
plt.grid(True, which='both', linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()