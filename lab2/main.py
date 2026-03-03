import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Початкові експериментальні дані
# ==========================================================

x_base = np.array([1000, 2000, 4000, 8000, 16000], dtype=float)
y_base = np.array([3, 5, 11, 28, 85], dtype=float)

# ==========================================================
# 2. Таблиця розділених різниць
# ==========================================================

def divided_differences(x, y):
    n = len(x)
    coef = np.zeros((n, n))
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])

    return coef

# ==========================================================
# 3. Поліном Ньютона
# ==========================================================

def newton_polynomial(x_data, coef, x):
    n = len(x_data)
    result = coef[0, 0]

    for i in range(1, n):
        term = coef[0, i]
        for j in range(i):
            term *= (x - x_data[j])
        result += term

    return result

# ==========================================================
# 4. Поліном Лагранжа
# ==========================================================

def lagrange_polynomial(x_data, y_data, x):
    n = len(x_data)
    result = 0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if j != i:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result

# ==========================================================
# 5. Факторіальний метод
# ==========================================================

def factorial_polynomial(x_data, y_data, x):
    h = x_data[1] - x_data[0]
    t = (x - x_data[0]) / h
    n = len(x_data)

    delta = np.zeros((n, n))
    delta[:, 0] = y_data

    for j in range(1, n):
        for i in range(n - j):
            delta[i, j] = delta[i+1, j-1] - delta[i, j-1]

    result = delta[0, 0]
    fact = 1
    mult = 1

    for i in range(1, n):
        mult *= (t - (i-1))
        fact *= i
        result += mult * delta[0, i] / fact

    return result

# ==========================================================
# 6. ПРОГНОЗ при n = 6000
# ==========================================================

coef_base = divided_differences(x_base, y_base)

prediction_newton = newton_polynomial(x_base, coef_base, 6000)
prediction_factorial = factorial_polynomial(x_base, y_base, 6000)

print("Прогноз Ньютона при n = 6000:", prediction_newton)
print("Прогноз факторіальний при n = 6000:", prediction_factorial)

# ==========================================================
# 7. Інтерполяція (5 вузлів) — ОДИН ГРАФІК
# ==========================================================

x_plot = np.linspace(1000, 16000, 500)
y_newton = np.array([newton_polynomial(x_base, coef_base, xi) for xi in x_plot])

plt.figure()
plt.scatter(x_base, y_base)
plt.plot(x_plot, y_newton)
plt.title("Інтерполяція (5 вузлів)")
plt.xlabel("n")
plt.ylabel("t (мс)")
plt.grid()
plt.show()

# ==========================================================
# >>> ДОДАНО: Похибка для 5 вузлів <<<
# ==========================================================

# Вважаємо 20 вузлів більш точною апроксимацією
x_20 = np.linspace(1000, 16000, 20)
y_20_nodes = np.array([newton_polynomial(x_base, coef_base, xi) for xi in x_20])
coef_20 = divided_differences(x_20, y_20_nodes)
y_20 = np.array([newton_polynomial(x_20, coef_20, xi) for xi in x_plot])

error_5 = np.abs(y_newton - y_20)

plt.figure()
plt.plot(x_plot, error_5)
plt.title("Абсолютна похибка (5 вузлів)")
plt.xlabel("n")
plt.ylabel("Похибка")
plt.grid()
plt.show()

# ==========================================================
# 8. Дослідження 1: Фіксований інтервал
# ==========================================================

for nodes in [10, 20]:
    x_nodes = np.linspace(1000, 16000, nodes)
    y_nodes = np.array([newton_polynomial(x_base, coef_base, xi) for xi in x_nodes])

    coef_nodes = divided_differences(x_nodes, y_nodes)
    y_interp = np.array([newton_polynomial(x_nodes, coef_nodes, xi) for xi in x_plot])

    plt.figure()
    plt.scatter(x_nodes, y_nodes)
    plt.plot(x_plot, y_interp)
    plt.title(f"Інтерполяція при {nodes} вузлах")
    plt.xlabel("n")
    plt.ylabel("t (мс)")
    plt.grid()
    plt.show()

    error = np.abs(y_newton - y_interp)

    plt.figure()
    plt.plot(x_plot, error)
    plt.title(f"Абсолютна похибка ({nodes} вузлів)")
    plt.xlabel("n")
    plt.ylabel("Похибка")
    plt.grid()
    plt.show()

# ==========================================================
# 9. Дослідження 2: Фіксований крок
# ==========================================================

intervals = [(1000, 8000), (1000, 12000), (1000, 16000)]

for a, b in intervals:
    x_nodes = np.linspace(a, b, 10)
    y_nodes = np.array([newton_polynomial(x_base, coef_base, xi) for xi in x_nodes])

    coef_nodes = divided_differences(x_nodes, y_nodes)
    y_interp = np.array([newton_polynomial(x_nodes, coef_nodes, xi) for xi in x_plot])

    plt.figure()
    plt.scatter(x_nodes, y_nodes)
    plt.plot(x_plot, y_interp)
    plt.title(f"Інтервал [{a}, {b}]")
    plt.xlabel("n")
    plt.ylabel("t (мс)")
    plt.grid()
    plt.show()

# ==========================================================
# 10. Ефект Рунге
# ==========================================================

def runge_function(x):
    return 1 / (1 + 25 * x**2)

x_dense = np.linspace(-1, 1, 500)
y_true = runge_function(x_dense)

for nodes in [5, 10, 20]:
    x_nodes = np.linspace(-1, 1, nodes)
    y_nodes = runge_function(x_nodes)

    coef_nodes = divided_differences(x_nodes, y_nodes)
    y_interp = np.array([newton_polynomial(x_nodes, coef_nodes, xi) for xi in x_dense])

    plt.figure()
    plt.plot(x_dense, y_true)
    plt.plot(x_dense, y_interp)
    plt.title(f"Ефект Рунге ({nodes} вузлів)")
    plt.grid()
    plt.show()

# ==========================================================
# 11. Порівняння з методом Лагранжа
# ==========================================================

y_lagrange = np.array([lagrange_polynomial(x_base, y_base, xi) for xi in x_plot])

plt.figure()
plt.scatter(x_base, y_base)
plt.plot(x_plot, y_newton, label="Ньютон")
plt.plot(x_plot, y_lagrange, linestyle="--", label="Лагранж")
plt.title("Порівняння: Ньютон vs Лагранж")
plt.xlabel("n")
plt.ylabel("t (мс)")
plt.legend()
plt.grid()
plt.show()

print("\nВСІ ДОСЛІДЖЕННЯ ЗАВЕРШЕНО")