"""
Main script for PSO Bird Nest Finder
Demonstrates Particle Swarm Optimization for finding optimal nest location
"""

import time
import numpy as np
from pso_algorithm import BirdNestPSO
from visualizations import visualize_pso, visualize_exhaustive_search, visualize_comparison


def main():
    print("=" * 60)
    print("PARTICLE SWARM OPTIMIZATION: BIRD NEST FINDER")
    print("=" * 60)
    
    # User inputs
    print("\nMap Size Options:")
    print("1. Small (200x200)")
    print("2. Large (800x800)")
    map_choice = input("Choose map size (1 or 2, default=1): ").strip() or "1"
    
    if map_choice == "2":
        map_size = 800
        n_predators = 8
        n_food = 8
    else:
        map_size = 200
        n_predators = 4
        n_food = 4
    
    n_particles = input(f"\nNumber of birds/particles (default=20): ").strip()
    n_particles = int(n_particles) if n_particles else 20
    
    n_iterations = input("Number of iterations (default=50): ").strip()
    n_iterations = int(n_iterations) if n_iterations else 50
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Map Size: {map_size}x{map_size}")
    print(f"  Particles: {n_particles}")
    print(f"  Predators: {n_predators}")
    print(f"  Food Sources: {n_food}")
    print(f"  Iterations: {n_iterations}")
    print(f"{'='*60}\n")
    
    # Initialize PSO
    print("Initializing PSO...")
    pso = BirdNestPSO(map_size=map_size, n_particles=n_particles, 
                      n_predators=n_predators, n_food=n_food)
    
    print(f"Initial best fitness: {pso.global_best_fitness:.2f}")
    
    # Run PSO computation (timed for performance measurement)
    print("\nRunning PSO computation...")
    pso_start = time.time()
    best_pos, best_fit, actual_iterations = pso.run(n_iterations=n_iterations)
    pso_time = time.time() - pso_start
    
    print(f"PSO computation completed in {pso_time:.3f}s")
    print(f"Completed {actual_iterations}/{n_iterations} iterations")
    print(f"PSO Best Position: ({pso.global_best[0]:.2f}, {pso.global_best[1]:.2f})")
    print(f"PSO Best Fitness: {pso.global_best_fitness:.2f}")
    
    # Calculate PSO evaluations
    pso_evaluations = n_particles + (n_particles * actual_iterations)
    print(f"Total PSO Evaluations: {pso_evaluations}")
    
    # Visualize PSO animation
    print("\nShowing PSO animation (visualization only)...")
    visualize_pso(pso, n_iterations=actual_iterations, interval=400, computation_time=pso_time)
    
    # Exhaustive search for comparison
    resolution = 100 if map_size == 200 else 150
    ex_start = time.time()
    best_pos, best_fit = visualize_exhaustive_search(pso, resolution=resolution)
    ex_time = time.time() - ex_start
    
    print(f"\nExhaustive Search Best Position: ({best_pos[0]:.2f}, {best_pos[1]:.2f})")
    print(f"Exhaustive Search Best Fitness: {best_fit:.2f}")
    print(f"Exhaustive Search Time: {ex_time:.3f}s")
    
    # Calculate accuracy metrics
    position_distance = np.linalg.norm(pso.global_best - best_pos)
    max_distance = np.linalg.norm([map_size, map_size])
    position_accuracy = max(0, (1 - position_distance / max_distance) * 100)
    
    # Display comparison
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"{'='*60}")
    print(f"PSO Computation Time: {pso_time:.3f}s | Exhaustive Time: {ex_time:.3f}s")
    if pso_time > 0:
        print(f"Speedup: {ex_time/pso_time:.2f}x faster with PSO")
    
    print(f"PSO Evaluations: {pso_evaluations} | Exhaustive: {resolution**2}")
    print(f"Evaluation Efficiency: {(resolution**2)/pso_evaluations:.2f}x fewer evaluations")
    print(f"Position Distance from Optimal: {position_distance:.2f}")
    print(f"Position Accuracy: {position_accuracy:.1f}%")
    print(f"{'='*60}\n")
    
    # Show visual comparison
    print("Displaying comparison visualization...")
    visualize_comparison(pso_time, ex_time, pso_evaluations, resolution, 
                        position_distance, position_accuracy, actual_iterations, n_iterations)


if __name__ == "__main__":
    main()
