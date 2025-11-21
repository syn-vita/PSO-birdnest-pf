import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time

class BirdNestPSO:
    def __init__(self, map_size=200, n_particles=20, n_predators=4, n_food=4):
        self.map_size = map_size
        self.n_particles = n_particles
        self.n_predators = n_predators
        self.n_food = n_food
        
        # PSO parameters
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (personal best) weight
        self.c2 = 1.5  # social (global best) weight
        
        # Initialize predators and food sources with good spacing
        self.predators = self._generate_spaced_points(n_predators)
        self.food_sources = self._generate_spaced_points(n_food)
        
        # Initialize particles
        self.particles = np.random.rand(n_particles, 2) * map_size
        self.velocities = np.random.randn(n_particles, 2) * 2
        
        # Personal best positions and fitness
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.array([self._fitness(p) for p in self.particles])
        
        # Global best
        best_idx = np.argmax(self.personal_best_fitness)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        self.history = []
        
        # Convergence tracking
        self.convergence_threshold = 0.01  # Stop if average movement < this
        self.convergence_iterations = 5  # Must be stable for this many iterations
        self.movement_history = []
        
    def _generate_spaced_points(self, n_points):
        """Generate points with good spacing to avoid clustering"""
        points = []
        min_distance = self.map_size / (2 * np.sqrt(n_points) + 1)
        
        attempts = 0
        max_attempts = 1000
        
        while len(points) < n_points and attempts < max_attempts:
            new_point = np.random.rand(2) * self.map_size
            
            # Check distance from existing points
            if len(points) == 0:
                points.append(new_point)
            else:
                distances = [np.linalg.norm(new_point - p) for p in points]
                if min(distances) > min_distance:
                    points.append(new_point)
            
            attempts += 1
        
        # If we couldn't generate enough points, fill the rest randomly
        while len(points) < n_points:
            points.append(np.random.rand(2) * self.map_size)
        
        return np.array(points)
    
    def _fitness(self, position):
        """
        Fitness = Distance to closest predator - Distance to furthest food
        Higher is better
        """
        # Distance to all predators
        predator_distances = [np.linalg.norm(position - pred) for pred in self.predators]
        closest_predator_dist = min(predator_distances)
        
        # Distance to all food sources
        food_distances = [np.linalg.norm(position - food) for food in self.food_sources]
        furthest_food_dist = max(food_distances)
        
        fitness = closest_predator_dist - furthest_food_dist
        return fitness
    
    def update(self):
        """Update particle positions and velocities"""
        # Track positions before update
        old_positions = self.particles.copy()
        
        for i in range(self.n_particles):
            # Update velocity
            r1, r2 = np.random.rand(2)
            
            # Inertia
            inertia = self.w * self.velocities[i]
            
            # Cognitive (memory vector - personal best)
            cognitive = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
            
            # Social (social vector - global best)
            social = self.c2 * r2 * (self.global_best - self.particles[i])
            
            # New velocity
            self.velocities[i] = inertia + cognitive + social
            
            # Limit velocity
            speed = np.linalg.norm(self.velocities[i])
            max_speed = self.map_size / 10
            if speed > max_speed:
                self.velocities[i] = (self.velocities[i] / speed) * max_speed
            
            # Update position
            self.particles[i] += self.velocities[i]
            
            # Boundary conditions (bounce back)
            for dim in range(2):
                if self.particles[i][dim] < 0:
                    self.particles[i][dim] = 0
                    self.velocities[i][dim] *= -0.5
                elif self.particles[i][dim] > self.map_size:
                    self.particles[i][dim] = self.map_size
                    self.velocities[i][dim] *= -0.5
            
            # Update personal best
            current_fitness = self._fitness(self.particles[i])
            if current_fitness > self.personal_best_fitness[i]:
                self.personal_best[i] = self.particles[i].copy()
                self.personal_best_fitness[i] = current_fitness
                
                # Update global best
                if current_fitness > self.global_best_fitness:
                    self.global_best = self.particles[i].copy()
                    self.global_best_fitness = current_fitness
        
        # Calculate average movement after all particles updated
        movements = np.linalg.norm(self.particles - old_positions, axis=1)
        avg_movement = np.mean(movements)
        self.movement_history.append(avg_movement)
        
        self.history.append({
            'particles': self.particles.copy(),
            'global_best': self.global_best.copy(),
            'global_best_fitness': self.global_best_fitness
        })
        
        return avg_movement
    
    def exhaustive_search(self, resolution=50):
        """Perform exhaustive search to find optimal nest locations"""
        x = np.linspace(0, self.map_size, resolution)
        y = np.linspace(0, self.map_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        fitness_map = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                position = np.array([X[i, j], Y[i, j]])
                fitness_map[i, j] = self._fitness(position)
        
        return X, Y, fitness_map
    
    def run(self, n_iterations=50):
        """Run PSO for n iterations with early stopping"""
        for iteration in range(n_iterations):
            avg_movement = self.update()
            
            # Check for convergence after minimum iterations
            if iteration >= self.convergence_iterations:
                recent_movements = self.movement_history[-self.convergence_iterations:]
                if all(m < self.convergence_threshold for m in recent_movements):
                    print(f"  Convergence detected at iteration {iteration + 1}")
                    print(f"  Average movement: {avg_movement:.4f} (threshold: {self.convergence_threshold})")
                    return self.global_best, self.global_best_fitness, iteration + 1
        
        return self.global_best, self.global_best_fitness, n_iterations

def visualize_pso(pso, n_iterations=50, interval=100, computation_time=None):
    """Animate PSO process"""
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
                                      edgecolors='orange', linewidths=2,
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
    
    # Store quiver object reference
    quiver_ref = {'obj': None}
    
    # Use history from pre-computed PSO run
    history_idx = {'current': 0}
    
    def init():
        particles_scatter.set_offsets(np.empty((0, 2)))
        global_best_scatter.set_offsets(np.empty((0, 2)))
        return particles_scatter, global_best_scatter, text
    
    def animate(frame):
        if frame < len(pso.history):
            history_data = pso.history[frame]
            particles = history_data['particles']
            global_best = history_data['global_best']
            global_best_fitness = history_data['global_best_fitness']
        else:
            particles = pso.particles
            global_best = pso.global_best
            global_best_fitness = pso.global_best_fitness
        
        # Update particles
        particles_scatter.set_offsets(particles)
        
        # Update global best
        global_best_scatter.set_offsets(global_best.reshape(1, -1))
        
        # Remove old quiver if it exists
        if quiver_ref['obj'] is not None:
            quiver_ref['obj'].remove()
        
        # Add new quiver arrows (use current velocities)
        quiver_ref['obj'] = ax.quiver(particles[:, 0], particles[:, 1],
                                       pso.velocities[:, 0], pso.velocities[:, 1],
                                       alpha=0.3, color='blue', scale=50, width=0.003)
        
        # Update text with computation time if available
        if computation_time is not None:
            text.set_text(f'Iteration: {frame}\n'
                          f'Best Fitness: {global_best_fitness:.2f}\n'
                          f'Best Position: ({global_best[0]:.1f}, {global_best[1]:.1f})\n'
                          f'Computation Time: {computation_time:.3f}s')
        else:
            text.set_text(f'Iteration: {frame}\n'
                          f'Best Fitness: {global_best_fitness:.2f}\n'
                          f'Best Position: ({global_best[0]:.1f}, {global_best[1]:.1f})')
        
        return particles_scatter, global_best_scatter, text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_iterations,
                         interval=interval, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def visualize_exhaustive_search(pso, resolution=100):
    """Visualize exhaustive search results"""
    print(f"\nPerforming exhaustive search with {resolution}x{resolution} = {resolution**2} evaluations...")
    start_time = time.time()
    
    X, Y, fitness_map = pso.exhaustive_search(resolution)
    
    exhaustive_time = time.time() - start_time
    
    # Find best position from exhaustive search
    best_idx = np.unravel_index(np.argmax(fitness_map), fitness_map.shape)
    best_position = np.array([X[best_idx], Y[best_idx]])
    best_fitness = fitness_map[best_idx]
    
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
    stats_text = (f'Exhaustive Search Time: {exhaustive_time:.2f}s\n'
                  f'Total Evaluations: {resolution**2}\n'
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
    
    return best_position, best_fitness, exhaustive_time

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
    
    # Summary box
    summary_text = "Summary: PSO efficiently found a near-optimal solution with significantly fewer evaluations"
    ax.text(0.5, 0.05, summary_text, ha='center', va='top', fontsize=12, 
            style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

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
    
    # Run PSO computation WITHOUT visualization (for accurate timing)
    print("\nRunning PSO computation...")
    pso_start = time.time()
    best_pos, best_fit, actual_iterations = pso.run(n_iterations=n_iterations)
    pso_time = time.time() - pso_start
    
    print(f"PSO computation completed in {pso_time:.3f}s")
    print(f"Completed {actual_iterations}/{n_iterations} iterations")
    print(f"PSO Best Position: ({pso.global_best[0]:.2f}, {pso.global_best[1]:.2f})")
    print(f"PSO Best Fitness: {pso.global_best_fitness:.2f}")
    print(f"Total PSO Evaluations: {n_particles + (n_particles * actual_iterations)}")
    
    # Now visualize the result (animation time doesn't affect comparison)
    print("\nShowing PSO animation (visualization only)...")
    visualize_pso(pso, n_iterations=actual_iterations, interval=100, computation_time=pso_time)
    
    # Exhaustive search
    resolution = 100 if map_size == 200 else 150
    best_pos, best_fit, ex_time = visualize_exhaustive_search(pso, resolution=resolution)
    
    print(f"\nExhaustive Search Best Position: ({best_pos[0]:.2f}, {best_pos[1]:.2f})")
    print(f"Exhaustive Search Best Fitness: {best_fit:.2f}")
    
    # Calculate accuracies
    position_distance = np.linalg.norm(pso.global_best - best_pos)
    max_distance = np.linalg.norm([map_size, map_size])
    position_accuracy = max(0, (1 - position_distance / max_distance) * 100)
    
    # Comparison (still print to console)
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"{'='*60}")
    print(f"PSO Computation Time: {pso_time:.3f}s | Exhaustive Time: {ex_time:.3f}s")
    if pso_time > 0:
        print(f"Speedup: {ex_time/pso_time:.2f}x faster with PSO")
    
    # Evaluations
    pso_evaluations = n_particles + (n_particles * actual_iterations)
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