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
    print("2. Large (1000x1000)")
    map_choice = input("Choose map size (1 or 2, default=1): ").strip() or "1"
    
    if map_choice == "2":
        map_size = 1000
        n_predators = 4
        n_food = 4
    else:
        map_size = 200
        n_predators = 4
        n_food = 4
    
    # Placement mode selection
    print("\nPlacement Mode:")
    print("1. Randomized (generated with spacing)")
    print("2. Fixed positions")
    placement_choice = input("Choose placement mode (1 or 2, default=1): ").strip() or "1"
    
    # Define fixed positions if selected
    fixed_predators = None
    fixed_food = None
    
    if placement_choice == "2":
        if map_size == 1000:
            # Fixed positions for 1000x1000 map
            fixed_predators = np.array([
                [299, 899],      # Top-left area
                [100, 349],     # Middle area
                [149, 103],     # Bottom-left area
                [500, 402]       # Top-right area
            ])
            fixed_food = np.array([
                [201, 799],     # Left side
                [802, 797],     # Top-right
                [203, 202],     # Bottom-left
                [801, 203]      # Bottom-right
            ])
        else:  # 200x200 map (scale down by factor of 5)
            fixed_predators = np.array([
                [59.8, 179.8],       # Top-left area
                [20, 69.8],      # Middle area
                [29.8, 20.6],      # Bottom-left area
                [100, 80.4]        # Top-right area
            ])
            fixed_food = np.array([
                [40.2, 159.8],       # Left side
                [160.4, 159.4],      # Top-right
                [40.6, 40.4],      # Bottom-left
                [160.2, 40.6]      # Bottom-right
            ])
    
    n_particles = input(f"\nNumber of birds/particles (default=9): ").strip()
    n_particles = int(n_particles) if n_particles else 9
    
    n_iterations = input("Number of iterations (default=40): ").strip()
    n_iterations = int(n_iterations) if n_iterations else 40
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Map Size: {map_size}x{map_size}")
    print(f"  Placement: {'Fixed' if placement_choice == '2' else 'Randomized'}")
    print(f"  Particles: {n_particles}")
    print(f"  Predators: {n_predators}")
    print(f"  Food Sources: {n_food}")
    print(f"  Iterations: {n_iterations}")
    print(f"{'='*60}\n")
    
    # Initialize PSO
    print("Initializing PSO...")
    pso = BirdNestPSO(map_size=map_size, n_particles=n_particles, 
                      n_predators=n_predators, n_food=n_food,
                      fixed_predators=fixed_predators, fixed_food=fixed_food)
    
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
    print(f"\nPerforming exhaustive search with {resolution}x{resolution} = {resolution**2} evaluations...")
    ex_start = time.time()
    X, Y, fitness_map = pso.exhaustive_search(resolution)
    ex_time = time.time() - ex_start
    
    # Find best position from exhaustive search
    best_idx = np.unravel_index(np.argmax(fitness_map), fitness_map.shape)
    best_pos = np.array([X[best_idx], Y[best_idx]])
    best_fit = fitness_map[best_idx]
    
    # Now visualize (this is separate from timing)
    print(f"Exhaustive search computation completed in {ex_time:.3f}s")
    print("\nDisplaying exhaustive search visualization...")
    from visualizations import visualize_exhaustive_search_display
    visualize_exhaustive_search_display(pso, X, Y, fitness_map, best_pos, best_fit, resolution)
    
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
