import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Создаем папку для графиков
os.makedirs('graphics', exist_ok=True)

# Загрузка данных
df = pd.read_csv("svm_results.csv")

# Настройка стиля
sns.set(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. Графики точности с учетом степени полинома
def plot_accuracy():
    fig = plt.figure(figsize=(15, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        ax = fig.add_subplot(1, 3, i)
        subset = df[df['Dataset'] == dataset]
        
        # Собираем все комбинации ядер и степеней
        kernels = []
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    kernels.append(('poly', degree))
            else:
                if not kernel_data.empty:
                    kernels.append((kernel, None))
        
        # Строим линии
        for kernel, degree in kernels:
            if kernel == 'poly':
                data = subset[(subset['Kernel'] == 'poly') & (subset['Degree'] == degree)]
                label = f'poly (d={degree})'
            else:
                data = subset[subset['Kernel'] == kernel]
                label = kernel.upper()
            
            sns.lineplot(
                x='C', 
                y='Test accuracy', 
                data=data,
                label=label,
                marker='o',
                ax=ax
            )
        
        ax.set_xscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel("Accuracy")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('graphics/accuracy_vs_c.png', dpi=300, bbox_inches='tight')

# 2. Heatmap для полиномиальных ядер
def plot_poly_heatmaps():
    plt.figure(figsize=(12, 5))
    for i, dataset in enumerate(['Moons', 'Circles' ], 1):
        plt.subplot(1, 2, i)
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

# 3. Анализ опорных векторов с учетом степени
def plot_sv_analysis():
    plt.figure(figsize=(15, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        plt.subplot(1, 3, i)
        subset = df[df['Dataset'] == dataset].copy()
        subset['Kernel_Degree'] = subset.apply(
            lambda x: f"{x['Kernel']} (d={x['Degree']})" if x['Kernel'] == 'poly' else x['Kernel'],
            axis=1
        )
        sns.scatterplot(
            x='Support_Vectors',
            y='Boundary_Vectors',
            hue='Kernel_Degree',
            style='Kernel_Degree',
            data=subset,
            s=100,
            palette='viridis'
        )
        plt.title(f"{dataset}: SV Analysis")
        plt.xlabel("Support Vectors")
        plt.ylabel("Boundary Vectors")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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

# 5. Графики для анализа опорных векторов (три отдельных)
def plot_support_vectors_count():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons']):
        ax = axes[i]
        subset = df[df['Dataset'] == dataset]
        
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    data = kernel_data[kernel_data['Degree'] == degree]
                    sns.lineplot(
                        x='C',
                        y='Support_Vectors',
                        data=data,
                        label=f'poly (d={degree})',
                        marker='o',
                        ax=ax
                    )
            else:
                if not kernel_data.empty:
                    sns.lineplot(
                        x='C',
                        y='Support_Vectors',
                        data=kernel_data,
                        label=kernel.upper(),
                        marker='o',
                        ax=ax
                    )
        
        ax.set_xscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel("Support Vectors")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('graphics/support_vectors_count.png', dpi=300, bbox_inches='tight')

def plot_boundary_vectors_count():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons']):
        ax = axes[i]
        subset = df[df['Dataset'] == dataset]
        
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    data = kernel_data[kernel_data['Degree'] == degree]
                    sns.lineplot(
                        x='C',
                        y='Boundary_Vectors',
                        data=data,
                        label=f'poly (d={degree})',
                        marker='o',
                        ax=ax
                    )
            else:
                if not kernel_data.empty:
                    sns.lineplot(
                        x='C',
                        y='Boundary_Vectors',
                        data=kernel_data,
                        label=kernel.upper(),
                        marker='o',
                        ax=ax
                    )
        
        ax.set_xscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel("Boundary Vectors")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('graphics/boundary_vectors_count.png', dpi=300, bbox_inches='tight')

def plot_boundary_ratio():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons']):
        ax = axes[i]
        subset = df[df['Dataset'] == dataset].copy()
        subset['Boundary_Ratio'] = subset['Boundary_Vectors'] / subset['Support_Vectors']
        
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    data = kernel_data[kernel_data['Degree'] == degree]
                    sns.lineplot(
                        x='C',
                        y='Boundary_Ratio',
                        data=data,
                        label=f'poly (d={degree})',
                        marker='o',
                        ax=ax
                    )
            else:
                if not kernel_data.empty:
                    sns.lineplot(
                        x='C',
                        y='Boundary_Ratio',
                        data=kernel_data,
                        label=kernel.upper(),
                        marker='o',
                        ax=ax
                    )
        
        ax.set_xscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel("Boundary Vectors Ratio")
        ax.set_ylim(0, 1)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('graphics/boundary_ratio.png', dpi=300, bbox_inches='tight')


# 1. Графики скорости работы с учетом степени полинома
def plot_time_dependence():
    fig = plt.figure(figsize=(15, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        ax = fig.add_subplot(1, 3, i)
        subset = df[df['Dataset'] == dataset]
        
        # Собираем все комбинации ядер и степеней
        kernels = []
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    kernels.append(('poly', degree))
            else:
                if not kernel_data.empty:
                    kernels.append((kernel, None))
        
        # Строим линии
        for kernel, degree in kernels:
            if kernel == 'poly':
                data = subset[(subset['Kernel'] == 'poly') & (subset['Degree'] == degree)]
                label = f'poly (d={degree})'
            else:
                data = subset[subset['Kernel'] == kernel]
                label = kernel.upper()
            
            sns.lineplot(
                x='C', 
                y='Fit_Time', 
                data=data,
                label=label,
                marker='o',
                ax=ax
            )
        
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel("Time (log scale)")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('graphics/time_vs_c.png', dpi=300, bbox_inches='tight')


def plot_time_vs_accuracy():
    fig = plt.figure(figsize=(15, 5))
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        ax = fig.add_subplot(1, 3, i)
        subset = df[df['Dataset'] == dataset]
        
        # Собираем все комбинации ядер и степеней
        kernels = []
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    kernels.append(('poly', degree))
            else:
                if not kernel_data.empty:
                    kernels.append((kernel, None))
        
        # Строим линии
        for kernel, degree in kernels:
            if kernel == 'poly':
                data = subset[(subset['Kernel'] == 'poly') & (subset['Degree'] == degree)]
                label = f'poly (d={degree})'
            else:
                data = subset[subset['Kernel'] == kernel]
                label = kernel.upper()
            
            sns.scatterplot(
                x='Test accuracy', 
                y='Fit_Time', 
                data=data,
                label=label,
                marker='o',
                ax=ax
            )
        ax.set_yscale('log')
        ax.set_title(dataset)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Time (log scale)")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('graphics/time_vs_accuracy.png', dpi=300, bbox_inches='tight')


from mpl_toolkits.mplot3d import Axes3D

def plot_3d_time_accuracy_c():
    fig = plt.figure(figsize=(18, 5))
    
    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        subset = df[df['Dataset'] == dataset]
        
        # Собираем все комбинации ядер и степеней
        kernels = []
        for kernel in ['linear', 'rbf', 'poly']:
            kernel_data = subset[subset['Kernel'] == kernel]
            if kernel == 'poly':
                degrees = kernel_data['Degree'].dropna().unique()
                for degree in degrees:
                    kernels.append(('poly', degree))
            else:
                if not kernel_data.empty:
                    kernels.append((kernel, None))
        
        # Создаем цветовую палитру для разных ядер
        colors = plt.cm.tab10(np.linspace(0, 1, len(kernels)))
        
        # Строим точки для каждой комбинации
        for idx, (kernel, degree) in enumerate(kernels):
            if kernel == 'poly':
                data = subset[(subset['Kernel'] == 'poly') & (subset['Degree'] == degree)]
                label = f'poly (d={degree})'
            else:
                data = subset[subset['Kernel'] == kernel]
                label = kernel.upper()
            
            # Преобразуем в логарифмическую шкалу
            log_c = np.log10(data['C'])
            log_time = np.log10(data['Fit_Time'])
            accuracy = data['Test accuracy']
            
            ax.scatter(
                log_c, 
                log_time, 
                accuracy,
                c=[colors[idx]],
                label=label,
                s=50,
                depthshade=True
            )
        
        ax.set_title(dataset)
        ax.set_xlabel("log C")
        ax.set_ylabel("log Time (s)")
        ax.set_zlabel("Accuracy")
        ax.legend()
        
        # Добавляем сетку для лучшей читаемости
        ax.xaxis._axinfo['grid']['color'] = (0.9, 0.9, 0.9, 0.9)
        ax.yaxis._axinfo['grid']['color'] = (0.9, 0.9, 0.9, 0.9)
        ax.zaxis._axinfo['grid']['color'] = (0.9, 0.9, 0.9, 0.9)
    
    plt.tight_layout()
    plt.savefig('graphics/3d_time_accuracy_c.png', dpi=300, bbox_inches='tight')

  

# Запуск всех визуализаций
# plot_accuracy()
# plot_poly_heatmaps()
# plot_sv_analysis()
# plot_rbf_heatmaps()
# plot_support_vectors_count()
# plot_boundary_vectors_count()
# plot_boundary_ratio()
# plot_time_dependence()
# plot_time_vs_accuracy()

plot_3d_time_accuracy_c()  

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_interactive_3d_plot():
    # Создаем subplot для трех датасетов
    fig = make_subplots(rows=1, cols=3, 
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=['Iris', 'Circles', 'Moons'],
                        horizontal_spacing=0.05)

    # Цветовая схема для разных ядер
    color_discrete_map = {
        'linear': 'blue',
        'rbf': 'red',
        'poly_2': 'green',
        'poly_3': 'purple',
        'poly_4': 'orange'
    }

    for i, dataset in enumerate(['Iris', 'Circles', 'Moons'], 1):
        subset = df[df['Dataset'] == dataset]
        
        # Добавляем данные для каждого типа ядра
        for kernel in subset['Kernel'].unique():
            kernel_data = subset[subset['Kernel'] == kernel]
            
            if kernel == 'poly':
                # Для полиномиального ядра учитываем степень
                for degree in kernel_data['Degree'].dropna().unique():
                    data = kernel_data[kernel_data['Degree'] == degree]
                    name = f'poly (d={int(degree)})'
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=np.log10(data['C']),
                            y=np.log10(data['Fit_Time']),
                            z=data['Test accuracy'],
                            mode='markers',
                            name=name,
                            marker=dict(
                                size=6,
                                color=color_discrete_map.get(f'poly_{int(degree)}', 'gray'),
                                opacity=0.8
                            ),
                            hovertemplate=
                            "<b>%{text}</b><br><br>" +
                            "log(C): %{x:.2f}<br>" +
                            "log(Time): %{y:.2f}<br>" +
                            "Accuracy: %{z:.2f}<extra></extra>",
                            text=[f"{kernel}, C={c:.2f}" for c in data['C']]
                        ),
                        row=1, col=i
                    )
            else:
                # Для других ядер
                fig.add_trace(
                    go.Scatter3d(
                        x=np.log10(kernel_data['C']),
                        y=np.log10(kernel_data['Fit_Time']),
                        z=kernel_data['Test accuracy'],
                        mode='markers',
                        name=kernel.upper(),
                        marker=dict(
                            size=6,
                            color=color_discrete_map.get(kernel, 'gray'),
                            opacity=0.8
                        ),
                        hovertemplate=
                        "<b>%{text}</b><br><br>" +
                        "log(C): %{x:.2f}<br>" +
                        "log(Time): %{y:.2f}<br>" +
                        "Accuracy: %{z:.2f}<extra></extra>",
                        text=[f"{kernel}, C={c:.2f}" for c in kernel_data['C']]
                    ),
                    row=1, col=i
                )

    # Настраиваем layout
    fig.update_layout(
        title_text="3D Visualization of SVM Performance",
        height=600,
        width=1500,
        scene1=dict(
            xaxis_title='log(C)',
            yaxis_title='log(Time)',
            zaxis_title='Accuracy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        scene2=dict(
            xaxis_title='log(C)',
            yaxis_title='log(Time)',
            zaxis_title='Accuracy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        scene3=dict(
            xaxis_title='log(C)',
            yaxis_title='log(Time)',
            zaxis_title='Accuracy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Сохраняем как HTML файл, который можно открыть в браузере
    fig.write_html("graphics/interactive_3d_plot.html")
    return fig

create_interactive_3d_plot()