import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Создаем папку для графиков
os.makedirs('graphics', exist_ok=True)

# Загрузка данных
df = pd.read_csv("svm_results.csv")

# Настройка стиля
sns.set(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. Графики точности с сохранением в graphics/
def plot_accuracy():
    fig = plt.figure(figsize=(15, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        ax = fig.add_subplot(1, 3, i)
        subset = df[df['Dataset'] == dataset]
        for kernel in ['linear', 'rbf', 'poly']:
            data = subset[subset['Kernel'] == kernel]
            if not data.empty:
                sns.lineplot(
                    x='C', 
                    y='Test accuracy', 
                    data=data,
                    label=kernel.upper(),
                    marker='o',
                    ax=ax
                )
        ax.set_xscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig('graphics/accuracy_vs_c.png', dpi=300, bbox_inches='tight')

# 2. Heatmap для полиномиальных ядер
def plot_poly_heatmaps():
    plt.figure(figsize=(12, 5))
    for i, dataset in enumerate(['Circles', 'Moons', 'Iris'], 1):
        plt.subplot(1, 3, i)
        subset = df[(df['Dataset'] == dataset) & (df['Kernel'] == 'poly')]
        pivot = subset.pivot_table(
            index='C',
            columns='Degree',
            values='Test accuracy',
            aggfunc='mean'
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={'label': 'Accuracy'},
            vmin=0.5,
            vmax=1.0
        )
        plt.title(f"{dataset}: Poly Kernel")
        plt.xlabel("Degree")
        plt.ylabel("C")
    plt.tight_layout()
    plt.savefig('graphics/poly_heatmaps.png', dpi=300, bbox_inches='tight')

# 3. Соотношение опорных и граничных векторов
def plot_sv_analysis():
    plt.figure(figsize=(15, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        plt.subplot(1, 3, i)
        subset = df[df['Dataset'] == dataset]
        sns.scatterplot(
            x='Support_Vectors',
            y='Boundary_Vectors',
            hue='Kernel',
            style='Kernel',
            data=subset,
            s=100,
            palette='viridis'
        )
        plt.title(f"{dataset}: SV Analysis")
        plt.xlabel("Support Vectors")
        plt.ylabel("Boundary Vectors")
    plt.tight_layout()
    plt.savefig('graphics/sv_analysis.png', dpi=300, bbox_inches='tight')

# 4. Heatmap для RBF ядра
def plot_rbf_heatmaps():
    plt.figure(figsize=(12, 5))
    for i, dataset in enumerate(['Circles', 'Moons'], 1):
        plt.subplot(1, 2, i)
        subset = df[(df['Dataset'] == dataset) & (df['Kernel'] == 'rbf')]
        pivot = subset.pivot_table(
            index='C',
            columns='Sigma',
            values='Test accuracy',
            aggfunc='mean'
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={'label': 'Accuracy'},
            vmin=0.4 if dataset == 'Circles' else 0.7,
            vmax=1.0
        )
        plt.title(f"{dataset}: RBF Kernel")
        plt.xlabel("Sigma")
        plt.ylabel("C")
    plt.tight_layout()
    plt.savefig('graphics/rbf_heatmaps.png', dpi=300, bbox_inches='tight')

def plot_support_vectors_analysis():
    # Создаем фигуру с тремя графиками
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons']):
        subset = df[df['Dataset'] == dataset]
        
        # График 1: Общее количество опорных векторов vs C
        ax1 = axes[0, i]
        for kernel in ['linear', 'rbf', 'poly']:
            data = subset[subset['Kernel'] == kernel]
            if not data.empty:
                sns.lineplot(
                    x='C',
                    y='Support_Vectors',
                    data=data,
                    label=kernel.upper(),
                    marker='o',
                    ax=ax1
                )
        ax1.set_xscale('log')
        ax1.set_title(f"{dataset}: Total Support Vectors")
        ax1.set_xlabel("C (log scale)")
        ax1.set_ylabel("Number of SVs")
        ax1.legend()
        
        # График 2: Количество связанных опорных векторов vs C
        ax2 = axes[1, i]
        for kernel in ['linear', 'rbf', 'poly']:
            data = subset[subset['Kernel'] == kernel]
            if not data.empty:
                sns.lineplot(
                    x='C',
                    y='Boundary_Vectors',
                    data=data,
                    label=kernel.upper(),
                    marker='o',
                    ax=ax2
                )
        ax2.set_xscale('log')
        ax2.set_title(f"{dataset}: Boundary Support Vectors")
        ax2.set_xlabel("C (log scale)")
        ax2.set_ylabel("Number of Boundary SVs")
        
        # График 3: Доля связанных векторов от общего числа
        ax3 = axes[2, i]
        for kernel in ['linear', 'rbf', 'poly']:
            data = subset[subset['Kernel'] == kernel]
            if not data.empty:
                # Вычисляем долю связанных векторов
                data = data.copy()
                data['Boundary_Ratio'] = data['Boundary_Vectors'] / data['Support_Vectors']
                sns.lineplot(
                    x='C',
                    y='Boundary_Ratio',
                    data=data,
                    label=kernel.upper(),
                    marker='o',
                    ax=ax3
                )
        ax3.set_xscale('log')
        ax3.set_title(f"{dataset}: Boundary SVs Ratio")
        ax3.set_xlabel("C (log scale)")
        ax3.set_ylabel("Boundary SVs / Total SVs")
        ax3.set_ylim(0, 1)  # Ограничиваем от 0 до 1 для пропорции
    
    plt.tight_layout()
    plt.savefig('graphics/support_vectors_detailed.png', dpi=300, bbox_inches='tight')

# Запуск всех визуализаций
plot_accuracy()
plot_poly_heatmaps()
plot_sv_analysis()
plot_rbf_heatmaps()
plot_support_vectors_analysis()