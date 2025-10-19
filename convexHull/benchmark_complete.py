import numpy as np
import matplotlib.pyplot as plt
import time
from convexHull import compute_convex_hull

def count_operations(coords):
    """Conta operações no Gift Wrapping"""
    if len(coords) < 3:
        return 0, len(coords), 0
    
    n = len(coords)
    operations = 0
    hull = []
    coords = [tuple(c) for c in coords]
    
    operations += n  # min scan
    leftmost_x = min([xcoord[0] for xcoord in coords])
    leftmost_point = [t for t in coords if leftmost_x == t[0]][0]
    
    current_point = leftmost_point
    max_iterations = n + 1
    iterations = 0
    
    while iterations < max_iterations:
        hull.append(current_point)
        operations += n
        
        next_point = None
        for p in coords:
            if p != current_point:
                next_point = p
                break
        
        if next_point is None:
            break
        
        for check_point in coords:
            operations += 1
            dx1 = next_point[0] - current_point[0]
            dy1 = next_point[1] - current_point[1]
            dx2 = check_point[0] - current_point[0]
            dy2 = check_point[1] - current_point[1]
            cross = dx1 * dy2 - dy1 * dx2
            if cross < 0:
                next_point = check_point
        
        current_point = next_point
        iterations += 1
        
        if current_point == hull[0]:
            break
    
    h = len(hull)
    return operations, n, h


def generate_random_points(n, min_val=25, max_val=950):
    """Gera pontos aleatórios uniformes (distribuição normal)"""
    coords = []
    points = np.random.randint(min_val, max_val, size=2*n)
    for i in range(0, len(points)-1, 2):
        coords.append([points[i], points[i+1]])
    return coords


def generate_circle_points(n, center=[500, 500], radius=400):
    """Gera pontos em círculo - PIOR CASO (todos na envoltória, h=n)"""
    import math
    points = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append([int(x), int(y)])
    return points


def generate_gaussian_cluster(n, center=[500, 500], std_dev=150):
    """Gera pontos com distribuição gaussiana - MELHOR CASO (poucos na envoltória)"""
    coords = []
    for _ in range(n):
        x = int(np.random.normal(center[0], std_dev))
        y = int(np.random.normal(center[1], std_dev))
        # Limita aos bounds da tela
        x = max(25, min(950, x))
        y = max(25, min(950, y))
        coords.append([x, y])
    return coords


def benchmark_distribution(n_values, num_trials=3):
    """Compara diferentes distribuições de pontos"""
    distributions = {
        'Uniforme': generate_random_points,
        'Círculo (pior caso)': lambda n: generate_circle_points(n),
        'Gaussiana (melhor caso)': lambda n: generate_gaussian_cluster(n)
    }
    
    results = {dist: {'times': [], 'ops': [], 'hs': [], 'ns': []} 
               for dist in distributions}
    
    for dist_name, generator in distributions.items():
        print(f"\n=== {dist_name} ===")
        for n in n_values:
            print(f"  n={n}...", end=' ')
            trial_times = []
            trial_ops = []
            trial_hs = []
            
            for _ in range(num_trials):
                coords = generator(n)
                
                # Tempo
                start = time.perf_counter()
                hull = compute_convex_hull(coords)
                end = time.perf_counter()
                trial_times.append(end - start)
                
                # Operações
                ops, _, h = count_operations(coords)
                trial_ops.append(ops)
                trial_hs.append(h)
            
            results[dist_name]['times'].append(np.mean(trial_times))
            results[dist_name]['ops'].append(np.mean(trial_ops))
            results[dist_name]['hs'].append(np.mean(trial_hs))
            results[dist_name]['ns'].append(n)
            print(f"OK (t={np.mean(trial_times)*1000:.2f}ms, h={np.mean(trial_hs):.1f})")
    
    return results


def benchmark_incremental(max_points=200, step=10, num_trials=3):
    """Testa crescimento incremental (adicionar pontos gradualmente)"""
    n_values = list(range(step, max_points + 1, step))
    times = []
    ops = []
    hs = []
    interior_points = []
    
    print("\n=== Crescimento Incremental ===")
    for n in n_values:
        print(f"  n={n}...", end=' ')
        trial_times = []
        trial_ops = []
        trial_hs = []
        trial_interior = []
        
        for _ in range(num_trials):
            coords = generate_random_points(n)
            
            start = time.perf_counter()
            hull = compute_convex_hull(coords)
            end = time.perf_counter()
            
            trial_times.append(end - start)
            
            operations, _, h = count_operations(coords)
            trial_ops.append(operations)
            trial_hs.append(h)
            trial_interior.append(n - h)  # pontos dentro da região
        
        times.append(np.mean(trial_times))
        ops.append(np.mean(trial_ops))
        hs.append(np.mean(trial_hs))
        interior_points.append(np.mean(trial_interior))
        print(f"OK (h={np.mean(trial_hs):.1f}, interior={np.mean(trial_interior):.1f})")
    
    return n_values, times, ops, hs, interior_points


def plot_all_results():
    """Gera todos os gráficos solicitados"""
    
    # 1. Crescimento incremental
    print("\n" + "="*50)
    print("BENCHMARK 1: Crescimento Incremental")
    print("="*50)
    n_inc, times_inc, ops_inc, hs_inc, interior_inc = benchmark_incremental(
        max_points=200, step=10, num_trials=3
    )
    
    # 2. Comparação de distribuições
    print("\n" + "="*50)
    print("BENCHMARK 2: Diferentes Distribuições")
    print("="*50)
    n_values_dist = [20, 50, 100, 150, 200]
    dist_results = benchmark_distribution(n_values_dist, num_trials=3)
    
    # Criar figura com 6 subplots (2x3)
    fig = plt.figure(figsize=(18, 12))
    
    # === GRÁFICO 1: Custo Computacional (Operações vs n) ===
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(n_inc, ops_inc, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Número de Pontos (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Operações', fontsize=12, fontweight='bold')
    ax1.set_title('1. Custo Computacional (Operações)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # === GRÁFICO 2: Pontos na Envoltória vs Total ===
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(n_inc, hs_inc, 's-', linewidth=2, markersize=8, color='#3498db', label='Envoltória (h)')
    ax2.plot(n_inc, interior_inc, '^-', linewidth=2, markersize=8, color='#2ecc71', label='Interior (n-h)')
    ax2.set_xlabel('Número de Pontos (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax2.set_title('2. Pontos na Envoltória vs Interior', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # === GRÁFICO 3: Crescimento do Tempo ===
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(n_inc, [t*1000 for t in times_inc], 'D-', linewidth=2, markersize=8, color='#9b59b6')
    ax3.set_xlabel('Número de Pontos (n)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('3. Crescimento do Tempo de Execução', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # === GRÁFICO 4: Comparação Tempo por Distribuição ===
    ax4 = plt.subplot(2, 3, 4)
    colors = {
        'Uniforme': '#e67e22', 
        'Círculo (pior caso)': '#e74c3c', 
        'Gaussiana (melhor caso)': '#27ae60'
    }
    for dist_name, data in dist_results.items():
        ax4.plot(data['ns'], [t*1000 for t in data['times']], 
                'o-', linewidth=2, markersize=8, label=dist_name, color=colors[dist_name])
    ax4.set_xlabel('Número de Pontos (n)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('4. Tempo por Distribuição', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # === GRÁFICO 5: Comparação Operações por Distribuição ===
    ax5 = plt.subplot(2, 3, 5)
    for dist_name, data in dist_results.items():
        ax5.plot(data['ns'], data['ops'], 
                's-', linewidth=2, markersize=8, label=dist_name, color=colors[dist_name])
    ax5.set_xlabel('Número de Pontos (n)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Operações', fontsize=12, fontweight='bold')
    ax5.set_title('5. Operações por Distribuição', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style='plain', axis='y')
    
    # === GRÁFICO 6: Tamanho da Envoltória por Distribuição ===
    ax6 = plt.subplot(2, 3, 6)
    for dist_name, data in dist_results.items():
        ax6.plot(data['ns'], data['hs'], 
                '^-', linewidth=2, markersize=8, label=dist_name, color=colors[dist_name])
    ax6.set_xlabel('Número de Pontos (n)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Pontos na Envoltória (h)', fontsize=12, fontweight='bold')
    ax6.set_title('6. Tamanho da Envoltória por Distribuição', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Gift Wrapping - Análise Completa de Complexidade', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('gift_wrapping_analise.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico salvo como 'gift_wrapping_analise.png'")
    plt.show()
    
    # Resumo estatístico
    print("\n" + "="*50)
    print("RESUMO ESTATÍSTICO")
    print("="*50)
    print(f"\nCrescimento Incremental (n={n_inc[-1]}):")
    print(f"  Tempo: {times_inc[-1]*1000:.2f}ms")
    print(f"  Operações: {ops_inc[-1]:,.0f}")
    print(f"  Pontos na envoltória: {hs_inc[-1]:.1f}")
    print(f"  Pontos no interior: {interior_inc[-1]:.1f}")
    
    print(f"\nComparação de Distribuições (n={n_values_dist[-1]}):")
    for dist_name, data in dist_results.items():
        print(f"  {dist_name}:")
        print(f"    Tempo: {data['times'][-1]*1000:.2f}ms")
        print(f"    Operações: {data['ops'][-1]:,.0f}")
        print(f"    h médio: {data['hs'][-1]:.1f}")


if __name__ == "__main__":
    plot_all_results()