# Particle Swarm Optimization: Bird Nest Finder

A Python implementation of Particle Swarm Optimization (PSO) algorithm that simulates birds finding an optimal nest location while avoiding all predators and staying close to all food sources.

## Overview

This project demonstrates PSO through an analogy where a swarm of birds collectively searches for the best nesting location. The optimal location maximizes distance from all predators while minimizing distance to all food sources.

## Features

- **PSO Visualization**: Real-time visualization of particle movement and convergence
- **Exhaustive Search Comparison**: Compare PSO efficiency against brute-force search
- **Customizable Parameters**: Adjust map size, particle count, and iterations
- **Fitness Landscape Visualization**: Heatmap showing the complete solution space
- **Performance Metrics**: Detailed comparison of computation time, evaluations, and accuracy

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation

```bash
pip install numpy matplotlib
```

## Usage

Run the main script:

```bash
python main.py
```

### Interactive Configuration

The program will prompt you for:

1. **Map Size**:

   - Small (200×200) - 4 predators, 4 food sources
   - Large (800×800) - 8 predators, 8 food sources

2. **Number of Particles**: Number of birds in the swarm (default: 20)

3. **Iterations**: Maximum optimization iterations (default: 50)

### Example Session

```
Choose map size (1 or 2, default=1): 1
Number of birds/particles (default=20): 20
Number of iterations (default=50): 50
```

## Algorithm Details

### PSO Parameters

- **Inertia Weight (w)**: 0.5 - Controls momentum of particles
- **Cognitive Weight (c1)**: 1.5 - Attraction to personal best position
- **Social Weight (c2)**: 1.5 - Attraction to global best position

### Fitness Function

```
Fitness = Distance to Closest Predator - Distance to Furthest Food
```

Higher fitness values indicate better nest locations.

### Velocity Update

```
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))
```

Where:

- `v(t)` = current velocity
- `x(t)` = current position
- `pbest` = personal best position
- `gbest` = global best position
- `r1, r2` = random values [0,1]

## Visualizations

### 1. PSO Animation

- **Blue dots**: Individual birds (particles) with smooth interpolated movement
- **Yellow star**: Current best nest location (50% transparent)
- **Red X**: Predators to avoid
- **Green squares**: Food sources
- Animation uses 8-step interpolation for fluid motion

### 2. Fitness Landscape

- **Heatmap**: Color-coded fitness values across entire map
- **Yellow star**: PSO solution
- **Cyan star**: Optimal solution from exhaustive search
- Shows relative performance and accuracy

### 3. Comparison Statistics

- Computation time comparison
- Evaluation efficiency
- Position accuracy
- Speedup metrics

## Performance

Typical results on a 200×200 map:

- **PSO Time**: ~0.1-0.3 seconds
- **Exhaustive Search Time**: ~2-5 seconds
- **Speedup**: 10-50× faster
- **Evaluation Efficiency**: 100× fewer evaluations
- **Position Accuracy**: 85-99%

## Convergence

The algorithm includes early stopping when:

- Average particle movement < 0.1 (threshold)
- Condition maintained for 3 consecutive iterations

This prevents unnecessary computation once the swarm has converged. The convergence parameters can be adjusted in `pso_algorithm.py`:

- Decrease threshold to 0.01 for tighter convergence (more iterations)
- Increase iterations to 5 for more stable stopping criteria

## Code Structure

The project is modularized into four main files for clarity and maintainability:

```
main.py                    # Entry point and user interface
├── User input handling
├── PSO initialization and execution
├── Performance timing and metrics
└── Visualization orchestration

pso_algorithm.py          # Core PSO implementation
├── BirdNestPSO Class
│   ├── __init__()           # Initialize parameters and positions
│   ├── _fitness()           # Evaluate nest location quality
│   ├── update()             # PSO velocity and position updates
│   ├── run()                # Main optimization loop with early stopping
│   └── exhaustive_search()  # Brute-force optimal solution

visualizations.py         # All plotting and animation functions
├── visualize_pso()          # Animate PSO process with interpolation
├── visualize_exhaustive_search()  # Show fitness landscape heatmap
└── visualize_comparison()   # Display performance metrics

utils.py                  # Helper utilities
└── generate_spaced_points() # Smart placement algorithm for predators/food
```

## Customization

### Adjust PSO Parameters

Modify `pso_algorithm.py` in the `BirdNestPSO.__init__()` method:

```python
self.w = 0.5   # Inertia: higher = more exploration
self.c1 = 1.5  # Cognitive: higher = stronger memory
self.c2 = 1.5  # Social: higher = stronger swarm effect
```

### Change Convergence Criteria

In `pso_algorithm.py`:

```python
self.convergence_threshold = 0.1  # Movement threshold (lower = tighter convergence)
self.convergence_iterations = 3   # Stability duration (higher = more stable)
```

### Adjust Animation Speed

In `main.py`, modify the `visualize_pso()` call:

```python
visualize_pso(pso, n_iterations=actual_iterations, interval=400)  # Higher = slower
```

### Modify Fitness Function

Edit the `_fitness()` method in `pso_algorithm.py` to implement different objectives:

```python
def _fitness(self, position):
    # Current implementation uses minimax approach:
    # Maximize distance to nearest predator
    # Minimize distance to furthest food
    # Your custom fitness calculation here
    return fitness_value
```

## Applications

This PSO implementation can be adapted for:

- **Optimization Problems**: Function minimization/maximization
- **Resource Allocation**: Facility location problems
- **Machine Learning**: Hyperparameter tuning

## Educational Value

This project demonstrates:

1. **Swarm Intelligence**: Emergent collective behavior
2. **Metaheuristic Algorithms**: Global optimization techniques
3. **Trade-offs**: Speed vs. accuracy in optimization
4. **Visualization**: Understanding algorithm behavior
5. **Performance Analysis**: Benchmarking optimization methods

## Limitations

- PSO may converge to local optima (not guaranteed global optimum)
- Performance depends on parameter tuning
- Best for continuous optimization problems
- Requires fitness function evaluation at each particle position

## License

This project is open source and available for educational purposes.

## Author

Created as an educational demonstration of Particle Swarm Optimization algorithms.

---

**Note**: Results may vary between runs due to random initialization. Multiple runs can help assess algorithm robustness.
