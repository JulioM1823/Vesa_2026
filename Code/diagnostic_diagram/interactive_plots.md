# Interactive k–ω Bokeh application

The Bokeh server application wraps the existing `DiagnosticDiagram` solver. It
does not carry a second copy of any dispersion relation. Every physical update
constructs a fresh model, solves the existing omega and vertical-wavenumber
polynomials, continuity-tracks the roots, verifies/stitches the `k_z=0`
boundaries, and updates the existing Bokeh data sources.

## Run

From the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/bokeh serve --show Code/diagnostic_diagram/bokeh_app
```

If the environment is already installed, only the final command is needed. The
application requires a Bokeh server; it is not a standalone HTML export because
all recalculation callbacks execute the Python solvers.

## Interface

The model tabs appear in this order:

1. **3D MAG Wave k–ω Diagram**
2. **Alfvén Mode k–ω Diagram**
3. **Slow/Fast Magnetoacoustic Mode k–ω Diagram**
4. **Bunte 1993 k–ω Diagram**
5. **MT1982 k–ω Diagram**
6. **MT1981 k–ω Diagram**

Each tab owns its controls, callback, cache key, status, and data sources. Tabs
are calculated lazily, so opening the app does not synchronously solve all six
models. Every adjustable physical or diagram variable is a bounded slider.
Slider `value` changes are debounced for 250 ms, so a pause while dragging
starts a recalculation without waiting for slider release. Changing the selected
field, warning mask, or f-mode visibility is immediate and does not rerun the
physics. Changes to `Δz` and to a model's f-mode-only `g` use the existing roots
and update only the postprocessing.

Every tracked `k_z` solution is displayed simultaneously: six panels for 3D
MAG, four for Slow/Fast, and two for each remaining model. A single solver call
updates the entire model grid; calculations are never repeated per panel. The
panels use linked axes and a shared color normalization. Their default layouts
are 3×2, 2×2, and 2×1, respectively.

The panels show the existing phase-difference diagnostic, with a shared
selectable view of `Re(k_z)` or `Im(k_z)` to inspect propagation and evanescence. Each plot
matches the pipeline's 7×7-inch/700×700-pixel square format, serif typography,
inward ticks, `k_h [Mm^-1]` and `ν [mHz]` labels, black boundary curves, and
`cmasher.fusion` phase palette with the fixed `[-80, 80] deg` normalization.
The optional vertical-wavenumber views use the pipeline's `hot` palette and
full finite-data limits. Hovering
reports `k_h`, `ν=ω/2π`, phase, both parts of `k_z`, the solution slot, and
numerical-warning state. Pan, wheel zoom, box zoom, reset, save, legend hiding,
and the optional gray f-mode reference are available. Ordinary wheel and
trackpad gestures scroll the page even while the pointer is over a plot; wheel
zoom is intentionally opt-in through each plot's toolbar. The f-mode is off by
default, matching the active pipeline plot.

## Active physical controls

| Model | Dispersion-relation controls | Diagram/reference controls |
|---|---|---|
| 3D MAG | `a` [km/s], `c_s` [km/s], `N` [rad/s], `ω_ac` [rad/s], `θ` [deg], `φ` [deg] | `Δz` [km], f-mode `g` [km/s²] |
| Alfvén | `a` [km/s], `θ` [deg], `φ` [deg] | `Δz` [km], f-mode `g` [km/s²] |
| Slow/Fast | `a` [km/s], `c_s` [km/s], `θ` [deg], `φ` [deg] | `Δz` [km], f-mode `g` [km/s²] |
| Bunte 1993 | `a` [km/s], `c_s` [km/s], `g` [km/s²], `γ`, `H` [km], `τ` [s] | `Δz` [km] |
| MT1982 | `c_s` [km/s], `g` [km/s²], `γ`, `H` [km], `N` [rad/s], `τ` [s] | `Δz` [km] |
| MT1981 | `c_s` [km/s], `H` [km], `N` [rad/s] | `Δz` [km], f-mode `g` [km/s²] |

For 3D MAG, every update recomputes the existing derived components
`a_x=a sin(theta) cos(phi)`, `a_y=a sin(theta) sin(phi)`, and
`a_z=a cos(theta)`. `N` and `omega_ac` remain independent inputs, matching the
current configuration semantics rather than silently deriving them when `c_s`,
`g`, or `H` changes.

The two positive slow/fast omega curves are displayed together on every
Slow/Fast solution panel. Their existing
whole-curve median frequencies are used only to name the lower curve “slow” and
the upper curve “fast”; the four tracked `k_z` sheets retain generic branch
labels because the existing solver does not attach permanent mode identities to
those sheets.

## Resolution and performance

The default 48×56 grid spans the notebook's existing domain,
`k_h=0…8 Mm^-1` and `ν=0…11 mHz`. A 64×76 balanced option and explicit 96×120
detailed option are available. The exact notebook 200×200 grid is accepted by
the adapter for offline validation, but is intentionally not a live UI option.

Representative complete recalculation times on the development arm64 machine
(Python 3.10, NumPy 1.24, SciPy 1.10) are:

| Model | 48×56 interactive | 200×200 notebook |
|---|---:|---:|
| 3D MAG | ~2.3 s | ~40.0 s |
| Alfvén | ~1.0 s | ~13.8 s |
| Slow/Fast | ~1.5 s | ~18.5 s |
| Bunte 1993 | ~1.2 s | ~14.7 s |
| MT1982 | ~0.9 s | ~14.0 s |
| MT1981 | ~1.1 s | ~14.2 s |

The high-order omega roots take only tens of milliseconds. Most time is spent in
the existing two-dimensional `k_z` root matching, branch identity refinement,
topology diagnostics, and boundary association. No model is skipped: the exact
48×56 calculations are responsive enough with lazy loading, debouncing, and an
LRU cache. Rendering every solution adds data-source updates but no additional
root solves. Detailed mode is intentionally slower and the busy status is
painted before its synchronous calculation starts.

## Numerical handling and known implementation limitations

- Controls exclude zero sound speed, nonpositive scale height/gravity, and exact
  `|theta|=90 deg` singularities. All values are finite and bounded before a
  solver is called.
- Nonfinite roots remain `NaN` and appear with the mapper's missing-data color.
  Only the known `omega=0` indeterminate phase row is displayed as zero, matching
  the existing notebook convention.
- A failed recalculation clears the affected tab and shows the exception; it
  never leaves stale data or breaks another model.
- Boundary overlays use only curves verified by the existing `k_z=0`
  association. A curve that disappears may have left the displayed 0–11 mHz
  interval rather than ceased to exist globally.
- Raw tracked `k_z` branch numbers can change with parameters or grid
  resolution because the existing tracker chooses a parameter-dependent seed.
  For presentation, each new calculation is matched to the previous panel stack
  using a robust full-sheet complex-`k_z` cost and one global branch-axis
  permutation. The same permutation is applied to phase, real/imaginary `k_z`,
  and stitched-boundary arrays. Exact model/parameter/resolution states are
  cached, so revisiting a state restores the identical panel ordering.
- At an exact root degeneracy, two coincident sheets have no mathematically
  unique individual identity. The application retains the prior deterministic
  slot order and reports the alignment as ambiguous rather than inventing a
  physical mode name.
- `B` and `epsilon` are configured/documented for Bunte 1993 but do not occur in
  the current polynomial coefficients. They are disclosed in the UI and are not
  exposed as fake-live controls. At exactly `a=0`, the existing implementation
  selects a distinct non-magnetic formula and may yield no verified overlays.
- In the current MT1982 implementation, the omega polynomial uses `g` but not
  `N`, while the `k_z` expression uses `N` but not `g`. Its default algebraic
  omega candidates fail the existing `k_z=0` verification, so the application
  reports them but does not draw them as valid boundaries.
- Large branch-identity uncertainty near genuine degeneracies/crossings is
  reported separately from failed or high-residual polynomial roots. Only the
  latter unresolved numerical-artifact mask is hidden by the default mask.

## Validation

Run the app-specific validation with:

```bash
.venv/bin/python -m pytest -q \
  tests/test_interactive_models.py \
  tests/test_interactive_bokeh.py
```

The tests compare every default adapter result with direct
`DiagnosticDiagram` calls on identical coordinates, including omega roots,
complete `k_z` root sets, phase postprocessing, and stitched-boundary support.
They also perturb every exposed parameter for every model, restore the defaults,
verify cache reuse and NC component derivation, validate slow/fast ordering,
reject invalid inputs, validate the Bokeh document, reuse source objects, and
confirm that a callback changes only its own model. Additional tests globally
permute complete solution sheets, verify that the original panel identities are
recovered, exercise exact-degeneracy tie handling, populate every simultaneous
panel with one solve, and confirm that returning to a cached state restores each
panel exactly.
