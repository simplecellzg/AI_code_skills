---
name: simulation-development-best-practices
description: Use when developing scientific simulation programs, encountering GPU race conditions, performance bottlenecks, numerical instabilities, convergence failures, or need systematic debugging strategies for parallel computing code.
---

# Scientific Simulation Development Best Practices

This skill distills proven patterns from multiple real-world simulation projects, covering GPU acceleration, 257x performance optimization, systematic debugging, and AI-assisted development workflows.

## When to Use This Skill

**Triggering scenarios:**
- GPU kernel producing extreme values (10^20+), race conditions, or incorrect results
- Performance bottleneck consuming 80%+ CPU time in specific functions
- Simulation not converging, residual stagnating, or oscillating
- Porting CPU code to GPU (CUDA, AMReX, etc.)
- Numerical instabilities: NaN/Inf, negative pressure, conservation law violations
- MPI/OpenMP parallel code debugging
- Need to achieve 10x+ performance improvement

## Core Principles

### 1. Profile Before Optimizing

**Never guess where the bottleneck is.** Profiling reveals truth.

```bash
# CPU profiling
gcc -pg -O2 program.c && ./program && gprof program gmon.out

# GPU profiling
nvprof --print-gpu-summary ./program
nsys profile --stats=true ./program
ncu --set full --target-processes all ./program
```

**Real case:** Function consuming 85.9% time (6.3 billion calls) → precompute lookup table → 66% speedup. Combined with buffer reuse, spatial filtering, and vectorization → **257x total speedup** (40min → 14sec).

### 2. GPU Asynchronous Execution - The Hidden Trap

**Critical insight:** GPU kernel launches return immediately. Subsequent CPU code may run before GPU completes.

```cpp
// ❌ WRONG - Reading unfinished GPU results
ParallelFor(bx, [=] GPU_DEVICE (int i,int j,int k) {
    result[i] = compute(data[i]);
});
use_result(result);  // ⚠️ May read garbage or partial results!

// ✅ CORRECT - Explicit synchronization
ParallelFor(bx, compute_kernel);
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();  // Wait for GPU to finish
#endif
use_result(result);  // Safe
```

**When synchronization is mandatory:**
1. After `FillBoundary()` before accessing ghost cells
2. After data transformation before dependent computation
3. After loops launching multiple kernels (common trap!)
4. Before host reads device data

**Real failure:** Missing sync caused 10^20+ extreme values at block boundaries. Pattern: pollution always at x=0.25, 0.5, 0.75 (block boundaries) → clear race condition signature.

### 3. Systematic Debugging - Don't Guess, Trace

Follow 4-phase methodology instead of random trial-and-error.

#### Phase 1: Root Cause Investigation

**Gather quantitative evidence:**
```bash
# Example: Detect anomalies with quantitative script
awk 'NR>4 && NF==8 {
    if ($3 > 10 || $3 < 0 || $3 != $3) {  # > threshold or negative or NaN
        print "Anomaly at x=" $1 ", y=" $2 ", rho=" $3
        count++
    }
} END {
    print "Total anomalies:", count+0
}' output.dat
```

**Don't accept** "looks wrong" - get numbers: how many cells? what magnitude? where located?

#### Phase 2: Pattern Analysis

**Find spatial/temporal patterns:**
- Anomalies at block boundaries → ghost cell issue
- Anomalies growing exponentially → unstable scheme
- Random distribution → memory corruption
- Specific direction (e.g., y-velocity non-zero in 1D-x problem) → data dependency issue

**Real case:** v-velocity = 0.0296 (should be 0) in 1D Sod shock → y-direction flux corrupted → traced to missing sync after reconstruction loop.

#### Phase 3: Hypothesis Testing

**One hypothesis, one test, quantitative metric:**

```markdown
Hypothesis Matrix:
| ID | Hypothesis | Test | Metric | Expected |
|----|-----------|------|--------|----------|
| H1 | Missing sync after cv2pv | Add sync | v_rms | Decrease 10x+ |
| H2 | FillBoundary race | Add sync | Block diff | < 1e-10 |
| H3 | Reconstruction loop race | Add sync | Symmetry | Restored |
```

Test one at a time. Record quantitative results:
```
Before Fix: v_rms = 0.169256
After Fix:  v_rms = 0.00354  (48x improvement ✅)
```

#### Phase 4: Implement and Verify

**Create regression tests immediately:**
```bash
#!/bin/bash
# test_fix_verification.sh
echo "=== Verifying fix effectiveness ==="

# Test 1: Extreme values
max_val=$(awk 'NR>4{if($3>max)max=$3}END{print max}' output.dat)
if (( $(echo "$max_val < 10" | bc -l) )); then
    echo "✅ Test 1 passed: max=$max_val"
else
    echo "❌ Test 1 failed: max=$max_val"
fi

# Test 2: v-velocity in 1D problem
v_rms=$(awk 'NR>4{sum+=$5^2;n++}END{print sqrt(sum/n)}' output.dat)
if (( $(echo "$v_rms < 0.01" | bc -l) )); then
    echo "✅ Test 2 passed: v_rms=$v_rms"
else
    echo "❌ Test 2 failed: v_rms=$v_rms"
fi
```

### 4. Test-Driven Development for Simulations

**Write tests first, then implement.**

```python
# Test conservation laws
def test_mass_conservation():
    sim = Simulation(nx=128, ny=64)
    m0 = sim.total_mass()
    sim.run(steps=1000, dt=0.001)
    m1 = sim.total_mass()
    assert abs(m1 - m0) / m0 < 1e-10, f"Mass error: {abs(m1-m0)/m0}"

# Test convergence order
def test_convergence_order():
    errors = []
    for nx in [32, 64, 128, 256]:
        sim = Simulation(nx=nx)
        sim.run(T_final=1.0)
        errors.append(sim.compute_error(exact_solution))

    order = log(errors[0]/errors[1]) / log(2.0)
    assert order > 1.9, f"Expected 2nd order, got {order}"
```

**Test categories:**
- Unit: Individual functions (flux, reconstruction, EOS)
- Integration: Module interactions (solver + boundary)
- Regression: Compare with stored reference solutions
- Physical: Conservation, entropy, positivity

### 5. Performance Optimization Workflow

**4-stage progressive optimization** (real case achieving 257x speedup):

#### Stage 0: Eliminate Waste (15% gain)
```cpp
// ❌ Allocating in hot loop
for (int i = 0; i < 1e8; i++) {
    double* temp = new double[N];  // 100M allocations!
    compute(temp);
    delete[] temp;
}

// ✅ Reuse buffer
double* buffer = new double[N];
for (int i = 0; i < 1e8; i++) {
    compute(buffer);  // Single allocation
}
delete[] buffer;
```

**Result:** 15% faster from removing 100M new/delete calls.

#### Stage 1: Precompute Expensive Functions (66% gain)

**Pattern:** Function called 6.3 billion times with limited input space → build lookup table.

```cpp
// 85.9% time in this function - called 6.3B times
double legendre_polynomial(int l, double theta) {
    // Recursive computation ~50 FLOPs
}

// Solution: Precompute table (2.2 MB)
class LegendreTable {
    std::vector<std::vector<double>> table_;  // [l][theta_idx]
    double theta_min_, theta_max_;
    int n_theta_;

public:
    LegendreTable(int l_max, int n_theta, double theta_min, double theta_max) {
        table_.resize(l_max + 1);
        n_theta_ = n_theta;
        theta_min_ = theta_min;
        theta_max_ = theta_max;

        for (int l = 0; l <= l_max; l++) {
            table_[l].resize(n_theta);
            for (int i = 0; i < n_theta; i++) {
                double theta = theta_min + i * (theta_max - theta_min) / (n_theta - 1);
                table_[l][i] = legendre_polynomial(l, theta);
            }
        }
    }

    double lookup(int l, double theta) const {
        // Linear interpolation
        double idx_real = (theta - theta_min_) / (theta_max_ - theta_min_) * (n_theta_ - 1);
        int idx = (int)idx_real;
        if (idx >= n_theta_ - 1) return table_[l][n_theta_ - 1];
        double frac = idx_real - idx;
        return table_[l][idx] * (1 - frac) + table_[l][idx + 1] * frac;
    }
};
```

**Trade-off:** 2.2 MB memory for 66% speedup. Worth it!

#### Stage 2: Spatial Filtering (92.9% gain)

**Pattern:** Computation applies to small region, but looping over entire domain.

```cpp
// ❌ Check all cells (100% of domain)
for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
        for (int k = 0; k < nz; k++) {
            if (distance(cell[i,j,k], perturbation_center) < R_cutoff) {
                apply_perturbation(i, j, k);
            }
        }
    }
}

// ✅ Precompute affected region (5% of domain)
struct Region { int i_min, i_max, j_min, j_max, k_min, k_max; };
Region compute_affected_region(Point center, double R_cutoff) {
    // Compute bounding box
    return {i_min, i_max, j_min, j_max, k_min, k_max};
}

Region region = compute_affected_region(center, R_cutoff);
for (int i = region.i_min; i <= region.i_max; i++) {
    for (int j = region.j_min; j <= region.j_max; j++) {
        for (int k = region.k_min; k <= region.k_max; k++) {
            if (distance(cell[i,j,k], center) < R_cutoff) {
                apply_perturbation(i, j, k);
            }
        }
    }
}
```

**Result:** Loop over 5% cells instead of 100% → 95% reduction in iterations.

#### Stage 3: Vectorization & Fusion (171x final)

**Pattern:** Eliminate function call overhead, flatten lookup, enable SIMD.

```cpp
// ❌ Nested function calls
for (...) {
    value = lookup_function(l, m, theta);  // Function call overhead
}

// ✅ Inline and vectorize
#pragma omp simd
for (int i = 0; i < n; i++) {
    // Inline computation, compiler can vectorize
    int idx = (int)((theta[i] - theta_min) * inv_dtheta);
    value[i] = table_flat[l * n_theta + idx];  // Flat array, better cache
}
```

**Combined result:** 40 minutes → 14 seconds = **171x speedup** (real production case).

## GPU-Specific Patterns

### Pattern 1: Race Condition Diagnosis

**Symptoms checklist:**
- [ ] Extreme values (10^20+, 10^100+)
- [ ] Values at block boundaries (e.g., x=0.25, 0.5, 0.75 for 4 blocks)
- [ ] Symmetry breaking (v≠0 in 1D problem)
- [ ] Exponential growth over time steps
- [ ] Problem disappears with single block

**Diagnostic strategy:**
```bash
# 1. Check spatial pattern
awk 'NR>4{print $1, $2, $3}' output.dat | \
    grep -E "0\.25|0\.50|0\.75"  # Block boundaries

# 2. Check symmetry in 1D problem (should have v=0)
awk 'NR>4{sum+=$5^2; n++}END{print "v_rms =", sqrt(sum/n)}' output.dat

# 3. Compare single-block vs multi-block
diff <(grep "^# " singleblock.dat) <(grep "^# " multiblock.dat)
```

**Common root causes:**
1. Missing sync after FillBoundary()
2. Missing sync after data transform (cv2pv, pv2cv)
3. Missing sync after reconstruction loop (when loop launches multiple kernels)
4. Reading device array immediately after kernel launch

**Fix pattern:**
```cpp
// Identify: where is data written by GPU?
ParallelFor(bx, [=] GPU_DEVICE (int i,int j,int k) {
    prm_array(i,j,k,n) = transform(cons_array(i,j,k,n));  // ← writes here
});

// Identify: where is data read?
ParallelFor(bx2, [=] GPU_DEVICE (int i,int j,int k) {
    flux = compute_flux(prm_array(i,j,k,n));  // ← reads here
});

// Solution: sync between write and read
ParallelFor(bx, write_kernel);
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();  // ← ADD THIS
#endif
ParallelFor(bx2, read_kernel);
```

### Pattern 2: Virtual Function Pointers on GPU

**Problem:** CUDA doesn't support calling virtual functions through host object pointers.

```cpp
// ❌ Doesn't work on GPU
class EOS {
public:
    virtual double compute_pressure(double rho, double e) = 0;
};

EOS* eos = new IdealGas(gamma);
ParallelFor(bx, [=] GPU_DEVICE (int i,int j,int k) {
    p = eos->compute_pressure(rho, e);  // ❌ Error 700: illegal memory access
});

// ✅ Solution 1: Pass parameters directly, use inline functions
namespace EOS_GPU {
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    Real compute_pressure(Real gamma, Real rho, Real e) {
        return (gamma - 1.0) * rho * e;
    }
}

Real gamma = 1.4;  // Capture to local variable
ParallelFor(bx, [=] GPU_DEVICE (int i,int j,int k) {
    p = EOS_GPU::compute_pressure(gamma, rho, e);  // ✅ Works
});

// ✅ Solution 2: Template-based dispatch
template<int EOSType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real compute_pressure(Real rho, Real e, Real gamma) {
    if constexpr (EOSType == IDEAL_GAS) {
        return (gamma - 1.0) * rho * e;
    } else if constexpr (EOSType == STIFFENED_GAS) {
        return (gamma - 1.0) * rho * e - gamma * p_inf;
    }
}
```

### Pattern 3: Static Members in Device Code

**Problem:** Host static variables cannot be accessed in device code.

```cpp
class Solver {
public:
    static int do_gravity;  // Host static member
};

// ❌ Won't work
ParallelFor(bx, [=] GPU_DEVICE (int i,int j,int k) {
    if (Solver::do_gravity) { ... }  // ❌ Undefined reference or wrong value
});

// ✅ Capture to local variable
int use_gravity = Solver::do_gravity;  // Capture before kernel
ParallelFor(bx, [=] GPU_DEVICE (int i,int j,int k) {
    if (use_gravity) { ... }  // ✅ Works
});
```

## Numerical Stability Patterns

### Pattern 4: Parameter Validation

**Real failure case:** Hardcoded `velocity_scale = 3.0` caused 50% integration error. Root cause: Gauss-Hermite quadrature range mismatch.

```julia
# ❌ Wrong - hardcoded constant
const velocity_scale = 3.0  # Where did this come from?

# Gauss-Hermite quadrature expects variable ~ N(0, 1)
# But molecular velocities ~ N(0, k*T/m)
# Mismatch causes 50% error in integration!

# ✅ Correct - physics-based derivation
const velocity_scale = sqrt(T_ref)  # From kinetic theory: v_thermal = sqrt(kT/m)
```

**Lesson:** Every parameter must have physical justification. Document units and derivation.

### Pattern 5: Time Scale Mismatch

**Real failure case:** IBM force relaxation time τ_IBM << flow time scale τ_c → numerical oscillation.

```julia
# Problem diagnosis
τ_IBM = 0.01  # Relaxation time
τ_c = u_ref / L_ref = 10.6  # Characteristic flow time
ratio = τ_IBM / τ_c = 0.00094  # << 1, too stiff!

# Oscillation in residual, never converges

# Solution: match time scales
τ_IBM = 0.1  # Increased 10x
ratio = τ_IBM / τ_c = 0.94  # O(1), resolved ✅
```

**Principle:** Time scales must be comparable. If ratio < 0.01 or > 100, expect trouble.

### Pattern 6: Conservation Monitoring

**Embed conservation checks:**
```cpp
// Monitor mass, momentum, energy every N steps
if (step % check_interval == 0) {
    double mass = compute_total_mass();
    double mass_error = abs(mass - initial_mass) / initial_mass;

    if (mass_error > 1e-10) {
        std::cerr << "WARNING: Mass conservation violated by "
                  << mass_error << " at step " << step << std::endl;
    }

    // Log for postprocessing
    log_file << step << " " << mass_error << " "
             << momentum_error << " " << energy_error << std::endl;
}
```

## AI-Assisted Workflow Integration

### Skill Composition Strategy

**For complex GPU debugging:**
```
1. brainstorming - Understand problem, explore fix approaches
2. systematic-debugging - 4-phase root cause analysis
3. test-driven-development - Write tests before fix
4. verification-before-completion - Verify fix effectiveness
```

**For performance optimization:**
```
1. writing-plans - Design multi-stage optimization plan
2. dispatching-parallel-agents - Parallel profile different components
3. test-driven-development - Benchmark tests before/after
4. verification-before-completion - Validate speedup and correctness
```

**For new feature:**
```
1. brainstorming - Requirements, design alternatives
2. writing-plans - Detailed implementation steps
3. test-driven-development - Tests → implement → refactor
4. requesting-code-review - Review before merge
```

### Effective Prompting for Simulation Code

**❌ Vague request:**
> "My code is slow, make it faster"

**✅ Specific request with context:**
> "Profiling shows `compute_legendre()` takes 85.9% time, called 6.3B times with l ∈ [0,20] and θ ∈ [0,π]. Can we precompute a lookup table? Show implementation with interpolation and estimate memory cost."

**❌ Unclear problem:**
> "GPU version doesn't work"

**✅ Diagnostic info:**
> "GPU version produces density = 3.4e27 at step 185, always at x≈0.25,0.50,0.75 (block boundaries). Single-block run is fine. Checked with cuda-memcheck, no errors. Suspect race condition in ghost cell filling. Show me where to add synchronization."

## Common Pitfalls

### Pitfall 1: Premature Optimization

**Wrong:** Optimize before profiling.
**Right:** Profile → identify 80% time → optimize that 20% code → verify speedup.

### Pitfall 2: Synchronization Overhead

**Wrong:** Add sync after every ParallelFor "to be safe".
```cpp
ParallelFor(bx1, kernel1); Gpu::streamSynchronize();  // ⚠️
ParallelFor(bx2, kernel2); Gpu::streamSynchronize();  // ⚠️
ParallelFor(bx3, kernel3); Gpu::streamSynchronize();  // ⚠️
// 3 syncs, but kernel2 doesn't depend on kernel1, kernel3 doesn't depend on kernel2
```

**Right:** Only sync when truly needed.
```cpp
ParallelFor(bx1, kernel1);  // Independent
ParallelFor(bx2, kernel2);  // Independent
Gpu::streamSynchronize();   // Sync once before using results
use_results();
```

### Pitfall 3: Treating Symptoms

**Wrong:** See NaN → add `if (isnan(x)) x = 0;` → hides root cause.
**Right:** Trace where NaN originates → fix computation → NaN never appears.

### Pitfall 4: Testing Only "It Runs"

**Insufficient:**
- Code compiles ✓
- Runs without crash ✓

**Required:**
- Passes unit tests ✓
- Conserves mass/momentum/energy to < 1e-10 ✓
- Matches analytical solution (if available) ✓
- Convergence order matches theory ✓
- Parallel version agrees with serial ✓

## Framework-Specific Patterns

### AMReX/AMReX-Hydro

```cpp
// Common pattern: MFIter over MultiFab
for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    auto const& array = mf.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
        // Computation
    });
}

// Don't forget sync after FillBoundary
mf.FillBoundary(geom.periodicity());
#ifdef AMREX_USE_GPU
    Gpu::streamSynchronize();  // ← Essential!
#endif
```

### MPI Domain Decomposition

```cpp
// Pattern: Exchange ghost cells between ranks
void exchange_ghost_cells(double* data, int nx_local, int ny, int nz) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    // Non-blocking send/recv
    MPI_Request req[4];
    MPI_Irecv(left_ghost, ny*nz, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(right_ghost, ny*nz, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(left_boundary, ny*nz, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(right_boundary, ny*nz, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}
```

## Verification Techniques

### Method of Manufactured Solutions

```cpp
// Create problem with known exact solution
double exact_solution(double x, double y, double t) {
    return sin(M_PI * x) * sin(M_PI * y) * exp(-2 * M_PI * M_PI * nu * t);
}

// Add source term to PDE so exact solution satisfies it
double source_term(double x, double y, double t) {
    double u = exact_solution(x, y, t);
    double du_dt = -2 * M_PI * M_PI * nu * u;
    double laplacian_u = -2 * M_PI * M_PI * u;
    return du_dt - nu * laplacian_u;  // For heat equation u_t = nu * Δu
}

// Verify: run simulation, compare with exact_solution, expect error → 0 as dx,dt → 0
```

### Convergence Study

```python
resolutions = [32, 64, 128, 256, 512]
errors = []

for nx in resolutions:
    sim = Simulation(nx=nx, T_final=1.0)
    sim.run()
    errors.append(sim.L2_error(exact_solution))

# Compute convergence order
orders = []
for i in range(len(errors) - 1):
    order = log(errors[i] / errors[i+1]) / log(2.0)
    orders.append(order)
    print(f"Order between {resolutions[i]} and {resolutions[i+1]}: {order:.2f}")

# For 2nd-order scheme, expect order ≈ 2.0
assert all(o > 1.9 for o in orders), "Convergence order below expected"
```

## Quick Reference

### GPU Debugging Checklist

```
Problem: Extreme values, wrong results, symmetry breaking

Diagnostics:
□ Run with single block - problem disappears?
□ Check values at block boundaries - pattern there?
□ Check variables that should be zero - are they?
□ Use cuda-memcheck - memory errors?
□ Compare CPU vs GPU output - where diverge?

Common fixes:
□ Add sync after FillBoundary()
□ Add sync after data transforms (cv2pv, etc.)
□ Add sync after loops with multiple kernels
□ Capture static members to local variables
□ Replace virtual functions with inline/template
```

### Performance Optimization Checklist

```
□ Profile first (gprof, nvprof, nsys)
□ Identify hotspot (>50% time)
□ Check if computation can be:
  □ Precomputed (lookup table)
  □ Cached (memoization)
  □ Filtered (spatial/temporal)
  □ Vectorized (SIMD)
  □ Parallelized (GPU/MPI/OpenMP)
□ Implement one optimization
□ Benchmark (must show >10% gain)
□ Verify correctness unchanged
□ Repeat for next hotspot
```

### Numerical Stability Checklist

```
□ Check for NaN/Inf in all outputs
□ Verify physical bounds (ρ>0, p>0, T>0)
□ Monitor conservation laws (Δm/m < 1e-10)
□ Check CFL condition (CFL < 0.5 for explicit)
□ Verify boundary conditions (symmetry, periodicity)
□ Test with analytical solutions
□ Perform convergence study (order matches theory)
□ Check parameter time scales (ratios ~ O(1))
```

---

**Philosophy:** Correctness first, performance second. Systematic diagnosis beats random guessing. Tests prevent regressions. Documentation enables collaboration.
