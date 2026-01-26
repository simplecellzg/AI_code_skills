---
name: simulation-development-best-practices
description: Use when developing, optimizing, or debugging scientific simulation programs (CFD, DSMC, molecular dynamics, etc.). Provides proven patterns for GPU programming, performance optimization, numerical stability, and AI-assisted development.
---

# Scientific Simulation Development Best Practices

This skill provides proven patterns and best practices for developing, optimizing, and debugging scientific simulation programs, based on real-world projects including GPU acceleration, performance optimization, and numerical stability.

## When to Use This Skill

Use this skill when:
- Developing or modifying scientific simulation code (CFD, DSMC, MD, FEM, etc.)
- Implementing GPU acceleration (CUDA, OpenCL, etc.)
- Optimizing performance bottlenecks
- Debugging numerical instabilities or convergence issues
- Porting CPU code to GPU
- Implementing parallel algorithms (MPI, OpenMP, CUDA)
- Working with AMReX or similar adaptive mesh frameworks

## Core Principles

### 1. Performance Profiling First

**Never optimize without profiling.** Use tools to identify actual bottlenecks.

```bash
# CPU profiling with gprof
gcc -pg -O2 program.c -o program
./program
gprof program gmon.out > analysis.txt

# GPU profiling with nvprof/Nsight
nvprof --profile-all-streams ./program
nsys profile --stats=true ./program
ncu --set full ./program
```

**Key patterns from CAPEX-GPU and RMI projects:**
- Legendre polynomial calculations consumed 85.9% of time → precompute with lookup tables
- Memory allocation in hot loops → use buffer reuse
- Repeated calculations → cache results

### 2. GPU Asynchronous Programming Awareness

**GPU kernel launches are asynchronous.** Always synchronize when needed.

```cpp
// ❌ WRONG: Assuming kernel completes immediately
ParallelFor(bx, kernel_func);
use_data(result);  // May read garbage!

// ✅ CORRECT: Explicit synchronization
ParallelFor(bx, kernel_func);
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
use_data(result);  // Safe to use
```

**Common synchronization points needed:**
- After `FillBoundary()` before using ghost cells
- After data transforms before dependent computations
- After loops launching multiple kernels
- Before host-device data transfers

**From CAPEX-GPU experience:** Missing synchronization caused race conditions that produced 10^20+ extreme values.

### 3. Test-Driven Development for Simulations

**Write tests before implementing features.** This is critical for numerical code.

```python
# Test 1: Verify conservation laws
def test_mass_conservation():
    initial_mass = compute_total_mass()
    run_simulation(steps=100)
    final_mass = compute_total_mass()
    assert abs(final_mass - initial_mass) < 1e-10

# Test 2: Verify convergence order
def test_convergence_order():
    errors = []
    for n in [64, 128, 256, 512]:
        errors.append(run_simulation(n))
    order = compute_convergence_order(errors)
    assert order > 1.9  # Expect 2nd order
```

**Test categories:**
- **Unit tests:** Individual functions (e.g., WENO reconstruction)
- **Integration tests:** Module interactions (e.g., flux + boundary)
- **Regression tests:** Compare against reference solutions
- **Physical tests:** Conservation laws, entropy conditions

### 4. Systematic Debugging Methodology

**Follow a structured debugging process, don't guess.**

**Phase 1: Root Cause Investigation**
1. Collect complete error information
2. Verify problem is reproducible
3. Check recent code changes
4. Add diagnostic output (quantitative, not qualitative)

**Phase 2: Pattern Analysis**
1. Identify spatial/temporal patterns
2. Compare working vs. broken configurations
3. Understand data dependencies

**Phase 3: Hypothesis Testing**
1. Form single, testable hypotheses
2. Create minimal reproducible examples
3. Verify with quantitative metrics

**Phase 4: Implementation**
1. Apply one fix at a time
2. Verify fix effectiveness
3. Add regression tests

**From UGKS-IBM project:** Systematic analysis revealed velocity_scale=3.0 caused 50% error; changing to sqrt(T_ref) fixed it.

### 5. Numerical Stability Best Practices

**Guard against common numerical issues.**

```cpp
// Check for NaN/Inf
inline bool is_valid(double x) {
    return !isnan(x) && !isinf(x);
}

// Check physical bounds
inline bool is_physics_valid(double rho, double p, double gamma) {
    return rho > 0 && p > 0 && is_valid(rho) && is_valid(p);
}

// Clamp to prevent overflow
double safe_divide(double a, double b) {
    if (fabs(b) < 1e-15) return 0.0;
    return a / b;
}
```

**Common issues and fixes:**
- **Negative pressure:** Apply positivity-preserving limiters
- **Division by zero:** Add small epsilon or clamp
- **Overflow in reconstruction:** Use characteristic variables
- **Loss of precision:** Use double precision for accumulations

## Common Problem Patterns and Solutions

### Pattern 1: GPU Race Conditions

**Symptoms:**
- Random failures or inconsistent results
- Extreme values (10^20+)
- Symmetry breaking in symmetric problems

**Diagnosis:**
```bash
# Use CUDA-MEMCHECK
cuda-memcheck ./program

# Add detailed logging with thread/block IDs
printf("[%d,%d] value=%.6e\n", blockIdx.x, threadIdx.x, value);
```

**Solutions:**
1. Add synchronization at data dependencies
2. Use atomic operations for shared writes
3. Ensure proper ghost cell filling
4. Check array bounds in kernels

### Pattern 2: Performance Bottleneck in Mathematical Functions

**Symptoms:**
- Profiling shows 80%+ time in math functions
- Same calculations repeated millions of times

**Solutions:**
```cpp
// Solution A: Precompute lookup tables
class LegendreTable {
    std::vector<std::vector<double>> table;
public:
    LegendreTable(int lmax, int n_samples) {
        // Precompute all values
        table.resize(lmax + 1);
        for (int l = 0; l <= lmax; l++) {
            table[l].resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                table[l][i] = compute_legendre(l, theta[i]);
            }
        }
    }
    double lookup(int l, double theta) {
        // Interpolate from table
    }
};

// Solution B: Buffer reuse
void compute_with_buffer(double* buffer) {
    // Reuse buffer instead of new/delete
    for (int i = 0; i < N; i++) {
        use_buffer(buffer, i);
    }
}
```

**From RMI project:** Precomputing Legendre polynomials (2.2 MB table) achieved 66% speedup.

### Pattern 3: Memory Access Inefficiency

**Symptoms:**
- Low memory bandwidth utilization
- Cache misses dominate profiling

**Solutions:**
```cpp
// Solution A: Coalesced memory access
__global__ void good_kernel(double* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = data[idx];  // Coalesced
    }
}

// Solution B: Shared memory for frequently accessed data
__global__ void optimized_kernel(double* global_data) {
    __shared__ double s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    s_data[tid] = global_data[...];
    __syncthreads();
    // Use s_data for fast access
}

// Solution C: Flatten data structures
// Use 1D arrays instead of nested vectors for GPU
```

### Pattern 4: Convergence Issues

**Symptoms:**
- Residual stagnates or diverges
- Oscillations in solution
- Slow convergence rate

**Diagnosis and Solutions:**

**Issue 1: Time step too large**
```cpp
// Check CFL condition
double compute_dt(double max_wave_speed, double dx) {
    double cfl = 0.5;  // Safe value
    return cfl * dx / max_wave_speed;
}
```

**Issue 2: Boundary condition errors**
```cpp
// Verify boundary consistency
void check_boundaries(double* data, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        // Check periodic boundaries
        assert(fabs(data[0][j] - data[nx-1][j]) < 1e-10);
    }
}
```

**Issue 3: Incorrect discretization**
- Verify stencil coefficients sum to 1 (for averaging)
- Check symmetry properties
- Compare with analytical solutions for simple cases

### Pattern 5: Incorrect Physical Parameters

**Symptoms:**
- Results don't match expectations
- Dimensional inconsistencies
- Unphysical values (negative density, etc.)

**From UGKS-IBM project example:**
```julia
# ❌ WRONG: Hardcoded velocity scale
const velocity_scale = 3.0  # Caused 50% error!

# ✅ CORRECT: Physics-based scale
const velocity_scale = sqrt(T_ref)  # Proper normalization
```

**Best practices:**
1. Use non-dimensionalization
2. Document all physical constants with units
3. Validate parameter ranges
4. Test with known analytical solutions

## Optimization Strategies

### Strategy 1: Space-Time Tradeoffs

**When computation is expensive, trade memory for speed.**

```cpp
// Example: Precompute expensive functions
class ExpensiveFunctionCache {
    std::unordered_map<std::pair<int,int>, double> cache;
public:
    double compute(int l, int m) {
        auto key = std::make_pair(l, m);
        if (cache.find(key) != cache.end()) {
            return cache[key];  // Cache hit
        }
        double value = expensive_computation(l, m);
        cache[key] = value;
        return value;
    }
};
```

**Tradeoff considerations:**
- Cache size vs. hit rate
- Memory bandwidth vs. computation
- L1/L2/L3 cache hierarchy

### Strategy 2: Algorithmic Optimization

**Better algorithms beat micro-optimizations.**

| Problem | Naive O(N²) | Optimized O(N log N) | Speedup |
|---------|--------------|---------------------|---------|
| N-body | Direct summation | FFT/Tree methods | 100-1000x |
| Sorting | Bubble sort | Quick/Merge sort | 100x |
| Search | Linear | Binary/Hash | 10-1000x |

### Strategy 3: Vectorization and SIMD

**Enable compiler auto-vectorization.**

```cpp
// Use restrict to promise no aliasing
void vectorized_kernel(const double* __restrict__ input,
                     double* __restrict__ output,
                     int n) {
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        output[i] = process(input[i]);
    }
}
```

### Strategy 4: Parallel Decomposition

**Choose appropriate parallel strategy.**

```cpp
// Strategy 1: Domain decomposition (MPI)
// Good for: Large problems, distributed memory
mpiexec -np 4 ./program

// Strategy 2: Shared memory (OpenMP)
// Good for: Multi-core systems, shared memory
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    process(i);
}

// Strategy 3: GPU acceleration (CUDA)
// Good for: Massive parallelism, regular patterns
kernel<<<grid, block>>>(data, N);
```

## AI-Assisted Development Workflow

### Phase 1: Requirements Clarification

Use the `brainstorming` skill to:
1. Understand the physical problem
2. Identify constraints (accuracy, performance, memory)
3. Explore multiple approaches
4. Select optimal solution

### Phase 2: Design and Planning

Use the `writing-plans` skill to:
1. Create detailed implementation plan
2. Break into small, testable tasks
3. Define verification criteria
4. Get user approval

### Phase 3: Test-Driven Implementation

Use the `test-driven-development` skill to:
1. Write failing tests first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Commit frequently

### Phase 4: Verification and Optimization

Use the `verification-before-completion` skill to:
1. Run full test suite
2. Profile for bottlenecks
3. Compare with reference solutions
4. Document performance characteristics

## Code Quality Checklist

Before committing simulation code, verify:

### Correctness
- [ ] All tests pass (unit, integration, regression)
- [ ] Physical conservation laws satisfied
- [ ] Boundary conditions correct
- [ ] No NaN/Inf in outputs
- [ ] Results match analytical solutions (if available)

### Performance
- [ ] Profiled and optimized hot paths
- [ ] Memory usage within limits
- [ ] Parallel efficiency > 70%
- [ ] Cache utilization reasonable

### Robustness
- [ ] Handles edge cases
- [ ] Graceful error handling
- [ ] Input validation
- [ ] Numerical stability guards

### Maintainability
- [ ] Clear variable names
- [ ] Comments explain physics, not code
- [ ] Modular design
- [ ] Documented interfaces

## Common Pitfalls to Avoid

### Pitfall 1: Premature Optimization

**Don't optimize before profiling.**
- Profile first to find real bottlenecks
- Optimize the 20% of code that takes 80% of time
- Keep code readable until performance is proven issue

### Pitfall 2: Ignoring Numerical Precision

**Double vs. Single precision:**
- Use double for accumulations and sensitive calculations
- Single may be sufficient for visualization
- Test both for your specific problem

### Pitfall 3: Hardcoding Parameters

**Make parameters configurable:**
```cpp
// ❌ BAD
const double gamma = 1.4;
const int max_iter = 1000;

// ✅ GOOD
struct Parameters {
    double gamma = 1.4;
    int max_iter = 1000;
    double tolerance = 1e-6;
};
```

### Pitfall 4: Insufficient Testing

**Test beyond "it runs":**
- Test convergence properties
- Test conservation laws
- Test boundary conditions
- Test extreme inputs
- Test parallel correctness

### Pitfall 5: Poor Documentation

**Document the physics, not just the code:**
```cpp
/**
 * @brief Compute WENO5 reconstruction
 *
 * Uses 5th-order Weighted Essentially Non-Oscillatory scheme
 * to reconstruct left and right states at cell interfaces.
 *
 * @param[in] f Cell-centered values (stencil of 5 points)
 * @param[out] fL Reconstructed left state
 * @param[out] fR Reconstructed right state
 *
 * @note The stencil should be: f[i-2], f[i-1], f[i], f[i+1], f[i+2]
 * @see Jiang & Shu, JCP 1996 for algorithm details
 */
```

## Project-Specific Patterns

### AMReX Projects

**Common patterns:**
- Use `MultiFab` for distributed data
- `FillBoundary()` requires synchronization on GPU
- `ParallelFor` for GPU kernels
- `MFIter` for loop over patches

**GPU synchronization pattern:**
```cpp
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
```

### MPI Projects

**Common patterns:**
- Domain decomposition with ghost cells
- `MPI_Send`/`MPI_Recv` for boundary exchange
- `MPI_Allreduce` for global operations
- Non-blocking communication for overlap

**Error handling:**
```cpp
int err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if (err != MPI_SUCCESS) {
    fprintf(stderr, "MPI error: %d\n", err);
    MPI_Abort(MPI_COMM_WORLD, err);
}
```

### CUDA Projects

**Common patterns:**
- Coalesced memory access
- Shared memory for intra-block communication
- Streams for concurrent operations
- Error checking with macros

**Error checking macro:**
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

## Performance Metrics to Track

Track these metrics throughout development:

| Metric | Target | Tool |
|--------|--------|------|
| Time to solution | Baseline | time command |
| Memory usage | < available | /usr/bin/time |
| Parallel efficiency | > 70% | Compare 1 vs N procs |
| Cache hit rate | > 80% | perf stat |
| Floating point intensity | High | hardware counters |

## Verification Methods

### Method 1: Method of Manufactured Solutions

Create a problem with known exact solution:
```cpp
// Manufactured solution: u(x,t) = sin(x) * exp(-t)
double exact_solution(double x, double t) {
    return sin(x) * exp(-t);
}

// Compute source term from PDE
double source_term(double x, double t) {
    // f = du/dt - L(u)
    return -sin(x)*exp(-t) - laplacian(exact_solution(x,t));
}
```

### Method 2: Convergence Study

```cpp
// Run at multiple resolutions
for (int n : {32, 64, 128, 256}) {
    double error = run_simulation(n);
    errors.push_back(error);
}

// Compute convergence order
double order = log(errors[0]/errors[1]) / log(2.0);
```

### Method 3: Conservation Monitoring

```cpp
// Monitor invariants
double initial_mass = compute_mass();
double initial_energy = compute_energy();

for (int step = 0; step < max_steps; step++) {
    advance();
    double mass_error = fabs(compute_mass() - initial_mass);
    double energy_error = fabs(compute_energy() - initial_energy);
    assert(mass_error < tolerance);
}
```

## Getting Help

When stuck, use this systematic approach:

1. **Check the physics:** Does the issue violate physical principles?
2. **Verify numerics:** Are discretizations correct? Check stencil coefficients.
3. **Profile performance:** Where is time actually spent?
4. **Examine data:** Plot intermediate results, look for patterns.
5. **Simplify:** Create minimal test case.
6. **Compare:** Working vs. broken versions.
7. **Document:** Record findings for future reference.

## References

- CUDA C Programming Guide
- AMReX Documentation
- MPI Standard
- Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
- LeVeque, "Finite Volume Methods for Hyperbolic Problems"

---

**Remember:** The goal is correct, verifiable, maintainable code. Performance optimization comes after correctness is established.
