import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Запит до API
# -------------------------------------------------

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]

n = len(results)
print("Кількість вузлів:", n)


# -------------------------------------------------
# 2. Запис у файл
# -------------------------------------------------
with open("route_data.txt", "w", encoding="utf-8") as f:
    f.write("№ | Latitude | Longitude | Elevation\n")
    for i, point in enumerate(results):
        f.write(f"{i} | {point['latitude']} | {point['longitude']} | {point['elevation']}\n")


# -------------------------------------------------
# 3. Кумулятивна відстань
# -------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
y = np.array([p["elevation"] for p in results])
x = np.zeros(n)
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)
x = np.array(distances)

print("\nТабуляція (відстань, висота):")
for i in range(n):
    print(f"{i:2d} | {x[i]:10.2f} | {y[i]:8.2f}")

# -------------------------------------------------
# 4. Перший графік висоти маршруту
# -------------------------------------------------
plt.figure()
plt.plot(x, y, marker='o')
plt.xlabel("Distance (m)")
plt.ylabel("Elevation (m)")
plt.title("Профіль висоти маршруту")
plt.grid()
plt.show()

# -------------------------------------------------
# 5. Кубічний сплайн
# -------------------------------------------------
h = np.diff(x)
A = np.zeros((n, n))
b = np.zeros(n)
A[0, 0] = 1
A[-1, -1] = 1
for i in range(1, n - 1):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])


# -------------------------------------------------
# 6. Метод прогонки для трьохдіагональної матриці
# -------------------------------------------------
def thomas_algorithm(a, b_diag, c, d):
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x_sol = np.zeros(n)
    c_prime[0] = c[0] / b_diag[0]
    d_prime[0] = d[0] / b_diag[0]
    for i in range(1, n):
        denom = b_diag[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom
    x_sol[-1] = d_prime[-1]
    for i in reversed(range(n - 1)):
        x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i + 1]
    return x_sol


a = np.zeros(n)
c = np.zeros(n)
for i in range(1, n - 1):
    a[i] = h[i - 1]
    c[i] = h[i]
M = thomas_algorithm(a, np.diag(A), c, b)
print("\nКоефіцієнти кубічного сплайна M:")
print(M)


# -------------------------------------------------
# 7. Функція сплайна
# -------------------------------------------------
def spline_eval(xi):
    for i in range(n - 1):
        if x[i] <= xi <= x[i + 1]:
            hi = h[i]
            return (
                    M[i] * (x[i + 1] - xi) ** 3 / (6 * hi) +
                    M[i + 1] * (xi - x[i]) ** 3 / (6 * hi) +
                    (y[i] - M[i] * hi ** 2 / 6) * (x[i + 1] - xi) / hi +
                    (y[i + 1] - M[i + 1] * hi ** 2 / 6) * (xi - x[i]) / hi
            )


# -------------------------------------------------
# 8. Гладкий графік сплайна
# -------------------------------------------------
xx = np.linspace(x[0], x[-1], 500)
yy = np.array([spline_eval(xi) for xi in xx])
plt.figure()
plt.plot(x, y, 'o', label='Вузли')
plt.plot(xx, yy, label='Кубічний сплайн')
plt.title("Кубічний сплайн маршруту")
plt.grid()
plt.legend()
plt.show()

# -------------------------------------------------
# 9. Характеристики маршруту
# -------------------------------------------------
print("Загальна довжина маршруту (м):", x[-1])
total_ascent = sum(max(y[i] - y[i - 1], 0) for i in range(1, n))
print("Сумарний набір висоти (м):", total_ascent)
total_descent = sum(max(y[i - 1] - y[i], 0) for i in range(1, n))
print("Сумарний спуск (м):", total_descent)
mass = 80
g = 9.81
energy = mass * g * total_ascent
print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy / 1000)
print("Енергія (ккал):", energy / 4184)

# -------------------------------------------------
# 10. Графіки з різною кількістю вузлів
# -------------------------------------------------
for num_nodes in [10, 15, 20]:
    idxs = np.linspace(0, n - 1, num_nodes, dtype=int)
    x_nodes = x[idxs]
    y_nodes = y[idxs]
    h_nodes = np.diff(x_nodes)

    A_nodes = np.zeros((num_nodes, num_nodes))
    b_nodes = np.zeros(num_nodes)
    A_nodes[0, 0] = 1
    A_nodes[-1, -1] = 1
    for i in range(1, num_nodes - 1):
        A_nodes[i, i - 1] = h_nodes[i - 1]
        A_nodes[i, i] = 2 * (h_nodes[i - 1] + h_nodes[i])
        A_nodes[i, i + 1] = h_nodes[i]
        b_nodes[i] = 6 * ((y_nodes[i + 1] - y_nodes[i]) / h_nodes[i] - (y_nodes[i] - y_nodes[i - 1]) / h_nodes[i - 1])

    a_nodes = np.zeros(num_nodes)
    c_nodes = np.zeros(num_nodes)
    for i in range(1, num_nodes - 1):
        a_nodes[i] = h_nodes[i - 1]
        c_nodes[i] = h_nodes[i]
    M_nodes = thomas_algorithm(a_nodes, np.diag(A_nodes), c_nodes, b_nodes)


    def spline_eval_nodes(xi):
        for i in range(num_nodes - 1):
            if x_nodes[i] <= xi <= x_nodes[i + 1]:
                hi = h_nodes[i]
                return (
                        M_nodes[i] * (x_nodes[i + 1] - xi) ** 3 / (6 * hi) +
                        M_nodes[i + 1] * (xi - x_nodes[i]) ** 3 / (6 * hi) +
                        (y_nodes[i] - M_nodes[i] * hi ** 2 / 6) * (x_nodes[i + 1] - xi) / hi +
                        (y_nodes[i + 1] - M_nodes[i + 1] * hi ** 2 / 6) * (xi - x_nodes[i]) / hi
                )


    xx_nodes = np.linspace(x_nodes[0], x_nodes[-1], 500)
    yy_nodes = np.array([spline_eval_nodes(xi) for xi in xx_nodes])

    plt.figure()
    plt.plot(x_nodes, y_nodes, 'o', label='Вузли')
    plt.plot(xx_nodes, yy_nodes, label=f'Сплайн {num_nodes} вузлів')
    plt.title(f'Кубічний сплайн з {num_nodes} вузлів')
    plt.grid()
    plt.legend()
    plt.show()

# -------------------------------------------------
# 11. Аналіз градієнта
# -------------------------------------------------
grad_full = np.gradient(yy, xx) * 100  # %
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))
steep_sections = np.sum(np.abs(grad_full) > 15)
print("Ділянки з крутизною > 15%:", steep_sections)

# -------------------------------------------------
# 12. Похибка наближення сплайна
# -------------------------------------------------
yy_nodes_interp = np.array([spline_eval(xi) for xi in x])
error = y - yy_nodes_interp
plt.figure()
plt.plot(x, y, 'o', label='Вихідні вузли')
plt.plot(xx, yy, label='Сплайн')
plt.plot(x, error, 'x', label='Похибка')
plt.title('Похибка сплайна')
plt.grid()
plt.legend()
plt.show()