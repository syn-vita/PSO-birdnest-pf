"""
Visualization functions for PSO Bird Nest Finder
Contains all UI and plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_pso(pso, n_iterations=50, interval=100, computation_time=None, interpolation_steps=8):
    """Animate PSO process with smooth interpolation"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot predators
    ax.scatter(pso.predators[:, 0], pso.predators[:, 1], 
               c='red', s=200, marker='x', linewidths=3, 
               label='Predators', zorder=5)
    
    # Plot food sources
    ax.scatter(pso.food_sources[:, 0], pso.food_sources[:, 1], 
               c='green', s=200, marker='s', 
               label='Food Sources', zorder=5)
    
    # Initialize particle scatter
    particles_scatter = ax.scatter([], [], c='blue', s=50, alpha=0.6, label='Birds')
    
    # Initialize global best marker
    global_best_scatter = ax.scatter([], [], c='yellow', s=300, marker='*', 
                                      edgecolors='orange', linewidths=2, alpha=0.5,
                                      label='Best Nest Location', zorder=6)
    
    ax.set_xlim(0, pso.map_size)
    ax.set_ylim(0, pso.map_size)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('PSO: Birds Finding Optimal Nest Location', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Text for iteration and fitness
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        particles_scatter.set_offsets(np.empty((0, 2)))
        global_best_scatter.set_offsets(np.empty((0, 2)))
        return particles_scatter, global_best_scatter, text
    
    def animate(frame):
        # Calculate which iteration and interpolation step we're on
        iteration = frame // interpolation_steps
        step = frame % interpolation_steps
        alpha = step / interpolation_steps  # Interpolation factor (0 to 1)
        
        if iteration < len(pso.history):
            # Get current and next iteration data
            current_data = pso.history[iteration]
            
            if iteration + 1 < len(pso.history):
                next_data = pso.history[iteration + 1]
                # Interpolate between current and next positions
                particles = (1 - alpha) * current_data['particles'] + alpha * next_data['particles']
                global_best = (1 - alpha) * current_data['global_best'] + alpha * next_data['global_best']
            else:
                # Last iteration, no interpolation
                particles = current_data['particles']
                global_best = current_data['global_best']
            
            global_best_fitness = current_data['global_best_fitness']
        else:
            particles = pso.particles
            global_best = pso.global_best
            global_best_fitness = pso.global_best_fitness
        
        # Update particles
        particles_scatter.set_offsets(particles)
        
        # Update global best
        global_best_scatter.set_offsets(global_best.reshape(1, -1))
        
        # Update text with computation time if available
        if computation_time is not None:
            text.set_text(f'Iteration: {iteration}\n'
                          f'Best Fitness: {global_best_fitness:.2f}\n'
                          f'Best Position: ({global_best[0]:.1f}, {global_best[1]:.1f})\n'
                          f'Computation Time: {computation_time:.3f}s')
        else:
            text.set_text(f'Iteration: {iteration}\n'
                          f'Best Fitness: {global_best_fitness:.2f}\n'
                          f'Best Position: ({global_best[0]:.1f}, {global_best[1]:.1f})')
        
        return particles_scatter, global_best_scatter, text
    
    # Total frames = iterations * interpolation_steps
    total_frames = n_iterations * interpolation_steps
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames,
                         interval=interval // interpolation_steps, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def visualize_exhaustive_search_display(pso, X, Y, fitness_map, best_position, best_fitness, resolution):
    """Visualize exhaustive search results (display only, computation already done)"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot fitness heatmap
    contour = ax.contourf(X, Y, fitness_map, levels=50, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Fitness Score')
    
    # Plot predators
    ax.scatter(pso.predators[:, 0], pso.predators[:, 1], 
               c='red', s=300, marker='x', linewidths=4, 
               label='Predators', zorder=5)
    
    # Plot food sources
    ax.scatter(pso.food_sources[:, 0], pso.food_sources[:, 1], 
               c='darkgreen', s=300, marker='s', 
               label='Food Sources', zorder=5, edgecolors='black', linewidths=2)
    
    # Plot PSO best position
    ax.scatter(pso.global_best[0], pso.global_best[1], 
               c='yellow', s=400, marker='*', 
               edgecolors='orange', linewidths=2,
               label=f'PSO Best (Fitness: {pso.global_best_fitness:.2f})', zorder=6)
    
    # Plot exhaustive search best position
    ax.scatter(best_position[0], best_position[1], 
               c='cyan', s=400, marker='*', 
               edgecolors='blue', linewidths=2,
               label=f'Exhaustive Best (Fitness: {best_fitness:.2f})', zorder=6)
    
    ax.set_xlim(0, pso.map_size)
    ax.set_ylim(0, pso.map_size)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Exhaustive Search: Complete Fitness Landscape', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Calculate distance between PSO and optimal position
    position_distance = np.linalg.norm(pso.global_best - best_position)
    max_distance = np.linalg.norm([pso.map_size, pso.map_size])
    position_accuracy = max(0, (1 - position_distance / max_distance) * 100)
    
    # Calculate fitness accuracy (how close fitness values are)
    fitness_range = np.max(fitness_map) - np.min(fitness_map)
    if fitness_range > 0:
        fitness_diff = abs(best_fitness - pso.global_best_fitness)
        fitness_accuracy = max(0, (1 - fitness_diff / fitness_range) * 100)
    else:
        fitness_accuracy = 100.0
    
    # Add text with statistics
    stats_text = (f'Exhaustive Search Evaluations: {resolution**2}\n'
                  f'PSO Fitness: {pso.global_best_fitness:.2f}\n'
                  f'Optimal Fitness: {best_fitness:.2f}\n'
                  f'Position Distance: {position_distance:.2f}\n'
                  f'Position Accuracy: {position_accuracy:.1f}%\n'
                  f'Fitness Quality: {fitness_accuracy:.1f}%')
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()


def visualize_exhaustive_search(pso, resolution=100):
    """
    DEPRECATED: Use separate computation and visualization instead.
    This function combines both and includes visualization time in the computation timer.
    Kept for backward compatibility.
    """
    print(f"\nPerforming exhaustive search with {resolution}x{resolution} = {resolution**2} evaluations...")
    
    X, Y, fitness_map = pso.exhaustive_search(resolution)
    
    # Find best position from exhaustive search
    best_idx = np.unravel_index(np.argmax(fitness_map), fitness_map.shape)
    best_position = np.array([X[best_idx], Y[best_idx]])
    best_fitness = fitness_map[best_idx]
    
    visualize_exhaustive_search_display(pso, X, Y, fitness_map, best_position, best_fitness, resolution)
    
    return best_position, best_fitness


def visualize_comparison(pso_time, ex_time, pso_evaluations, resolution, position_distance, position_accuracy, actual_iterations, n_iterations):
    """Display comparison statistics in a visual window"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Title
    title_text = "PSO vs Exhaustive Search Comparison"
    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=20, fontweight='bold')
    
    # Time comparison
    time_text = "COMPUTATION TIME"
    ax.text(0.5, 0.85, time_text, ha='center', va='top', fontsize=16, fontweight='bold', color='darkblue')
    ax.text(0.5, 0.80, f"PSO: {pso_time:.3f}s  |  Exhaustive: {ex_time:.3f}s", 
            ha='center', va='top', fontsize=14)
    
    speedup = ex_time / pso_time if pso_time > 0 else 0
    speedup_color = 'green' if speedup > 1 else 'red'
    ax.text(0.5, 0.75, f"Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} with PSO", 
            ha='center', va='top', fontsize=14, fontweight='bold', color=speedup_color)
    
    # Evaluation comparison
    eval_text = "EVALUATIONS"
    ax.text(0.5, 0.65, eval_text, ha='center', va='top', fontsize=16, fontweight='bold', color='darkblue')
    ax.text(0.5, 0.60, f"PSO: {pso_evaluations:,}  |  Exhaustive: {resolution**2:,}", 
            ha='center', va='top', fontsize=14)
    
    efficiency = (resolution**2) / pso_evaluations
    ax.text(0.5, 0.55, f"PSO used {efficiency:.2f}x fewer evaluations", 
            ha='center', va='top', fontsize=14, fontweight='bold', color='green')
    
    # Iterations
    iter_text = "ITERATIONS"
    ax.text(0.5, 0.45, iter_text, ha='center', va='top', fontsize=16, fontweight='bold', color='darkblue')
    ax.text(0.5, 0.40, f"Completed: {actual_iterations}/{n_iterations} iterations", 
            ha='center', va='top', fontsize=14)
    
    if actual_iterations < n_iterations:
        ax.text(0.5, 0.35, "Early convergence detected", 
                ha='center', va='top', fontsize=14, fontweight='bold', color='green')
    
    # Accuracy
    accuracy_text = "ACCURACY"
    ax.text(0.5, 0.25, accuracy_text, ha='center', va='top', fontsize=16, fontweight='bold', color='darkblue')
    ax.text(0.5, 0.20, f"Position Distance from Optimal: {position_distance:.2f}", 
            ha='center', va='top', fontsize=14)
    
    accuracy_color = 'green' if position_accuracy >= 90 else 'orange' if position_accuracy >= 70 else 'red'
    ax.text(0.5, 0.15, f"Position Accuracy: {position_accuracy:.1f}%", 
            ha='center', va='top', fontsize=14, fontweight='bold', color=accuracy_color)
    
    plt.tight_layout()
    plt.show()
