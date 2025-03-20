import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Функция для визуализации решений SVC (с разделяющей линией, линиями margin и опорными векторами)
def plot_decision_boundary_sklearn(model, X, y, ax=None):
    if ax is None:
        ax = plt.gca()
    # Отображение точек данных
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.6)
    
    # Создаем сетку для отрисовки границы
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Вычисляем значение функции принятия решения
    Z = model.decision_function(grid).reshape(xx.shape)
    
    # Основная разделяющая линия: f(x)=0
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='-')
    
    # Линии, соответствующие margin: f(x)=+1 и f(x)=-1
    ax.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linewidths=1.5, linestyles='--')
    
    # Отображаем опорные векторы
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='g', label='Support Vectors')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"SVC (kernel={model.kernel})")
    ax.legend()

# Генерируем линейно разделимые данные с помощью make_blobs
X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.8, random_state=42)
# Приводим метки к ±1
y = np.where(y == 0, -1, 1)

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Задаем набор "снимков": количество обучающих примеров, на которых будем тренировать SVM
snapshot_indices = [10, 50, 100, len(X_train)]

# Определяем список ядер для сравнения
kernels = ['linear', 'rbf', 'poly']

# Для каждого количества обучающих примеров (snapshot) обучаем SVC с каждым ядром
for snapshot in snapshot_indices:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Обучение SVC на {snapshot} примерах", fontsize=16)
    
    # Берем первые snapshot примеров из обучающей выборки
    X_train_subset = X_train[:snapshot]
    y_train_subset = y_train[:snapshot]
    
    for idx, kernel in enumerate(kernels):
        # Для полиномиального ядра можно задать degree и coef0; здесь используются значения по умолчанию
        model = SVC(kernel=kernel, C=1.0)
        model.fit(X_train_subset, y_train_subset)
        # Вычисляем точность на тестовой выборке
        acc = model.score(X_test, y_test)
        # Отрисовываем решение на подграфике
        ax = axes[idx]
        plot_decision_boundary_sklearn(model, X_train_subset, y_train_subset, ax=ax)
        ax.set_title(f"{kernel.capitalize()} kernel\nТочность: {acc:.2f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
