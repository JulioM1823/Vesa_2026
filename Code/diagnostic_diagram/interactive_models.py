'''
Purpose
-------
Bokeh-facing adapters for the existing dispersion-relation solvers.

This module deliberately contains no Bokeh imports.  It translates user-facing
values into the parameter dictionaries already consumed by
``DiagnosticDiagram`` and packages solver output for a server application.
The dispersion relations themselves remain in :mod:`functions`.

Inputs
------
None
    Model defaults are read from config at import time.

Outputs
-------
Model definitions, parameter adapters, and diagram calculation helpers
    Solver parameters use km, s, and radians; display axes use Mm^-1 and mHz.
'''

from __future__ import annotations

# Import the metadata containers, caching helpers, and numerical libraries
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations
from math import pi
from time import perf_counter
from types import MappingProxyType
from typing import Mapping
import warnings

import numpy as np

try:  # Support both package imports and the notebook-style sys.path setup.
    from . import config
    from .numeric_controls import validate_parameter
    from .magnetic_geometry import components
    from .functions import DiagnosticDiagram
except ImportError:  # pragma: no cover - used by ``bokeh serve`` directory apps.
    import config
    from numeric_controls import validate_parameter
    from magnetic_geometry import components
    from functions import DiagnosticDiagram


# Define solver-to-display conversions and the default calculation domain
RAD_S_TO_MHZ = 1e3 / (2 * pi)
INV_KM_TO_INV_MM = 1e3
DEFAULT_DOMAIN = (0.0, 8.0, 0.0, 11.0)  # kh [Mm^-1], nu [mHz]
DEFAULT_RESOLUTION = (48, 56)  # (n_kh, n_omega)


@dataclass(frozen=True)
class ParameterDefinition:
    '''
    Purpose
    -------
    Description and safe UI range for one independently varied value.

    Inputs
    ------
    key, label, unit: str
        Solver/UI identifier, display label, and displayed physical unit.
    default, minimum, maximum, step: float
        Initial value and suggested slider interval/increment; physical validation
        is performed separately and can accept values outside the slider interval.
    role: str, optional
        'dispersion' requires a new root solve; 'diagram' changes postprocessing.
    help_text: str, optional
        Supplemental description of the parameter.

    Outputs
    -------
    definition: ParameterDefinition
        Frozen control metadata; title combines label and unit.
    '''

    # Store the control identity, units, and suggested numerical settings
    key: str
    label: str
    unit: str
    default: float
    minimum: float
    maximum: float
    step: float
    role: str = "dispersion"
    help_text: str = ""

    @property
    def title(self) -> str:
        '''
        Purpose
        -------
        Return the display label with its unit when one is defined.

        Inputs
        ------
        None
            Uses the label and unit stored on this definition.

        Outputs
        -------
        title: str
            Label with a bracketed unit, or the label alone when unit is empty.
        '''

        # Append physical units only when the parameter declares them
        return f"{self.label} [{self.unit}]" if self.unit else self.label


@dataclass(frozen=True)
class ModelDefinition:
    '''
    Purpose
    -------
    Describe a solver model and the parameters exposed by its controls.

    Inputs
    ------
    key, title, reference: str
        Configuration identifier, tab title, and model reference text.
    parameters: tuple of ParameterDefinition
        Ordered parameter controls exposed by this model.
    metadata_note: str, optional
        Implementation information displayed alongside the controls.

    Outputs
    -------
    definition: ModelDefinition
        Frozen model metadata with an immutable defaults mapping.
    '''

    # Store the model identity, ordered controls, and implementation metadata
    key: str
    title: str
    reference: str
    parameters: tuple[ParameterDefinition, ...]
    metadata_note: str = ""

    @property
    def defaults(self) -> Mapping[str, float]:
        '''
        Purpose
        -------
        Return an immutable mapping of parameter keys to their defaults.

        Inputs
        ------
        None
            Reads the ordered parameter definitions on this model.

        Outputs
        -------
        defaults: Mapping[str, float]
            Read-only mapping of all exposed parameter keys to their initial values.
        '''

        # Expose defaults without allowing callers to modify this mapping
        return MappingProxyType({item.key: item.default for item in self.parameters})


@dataclass(frozen=True)
class DiagramResult:
    '''
    Purpose
    -------
    Immutable-by-convention arrays ready for an interactive diagram.

    Inputs
    ------
    model_key: str
        Configuration identifier for the solved model.
    parameter_values: tuple of (str, float)
        Normalized parameter values used by the solve.
    kh_mm_inv, nu_mhz: np.ndarray
        One-dimensional horizontal-wavenumber (Mm^-1) and frequency (mHz) axes.
    phase_deg, kz_real, kz_imag: np.ndarray
        Arrays with shape (n_omega, n_kh, n_branch); phase is in degrees and
        vertical-wavenumber components are in km^-1.
    numerical_warning_mask, branch_identity_uncertain_mask: np.ndarray, bool
        Per-cell masks with shape (n_omega, n_kh).
    omega_boundaries_mhz, associated_boundaries_mhz: np.ndarray
        Boundary frequencies with shapes (n_kh, n_boundary) and
        (n_kh, n_branch, n_boundary), respectively; unavailable points are NaN.
    fmode_nu_mhz: np.ndarray
        One-dimensional f-mode reference frequency along kh_mm_inv (mHz).
    branch_labels, boundary_labels, warnings: tuple of str
        Display names and calculation-quality messages.
    summary: Mapping
        Read-only calculation dimensions, coverage, residuals, and status counts.
    elapsed_seconds: float
        Wall-clock duration of the calculation or postprocessing update (s).
    propagating_mask, vertical_group_velocity_km_s: np.ndarray or None
        Optional boolean propagation flags and vertical group velocities (km/s),
        each with returned-root shape (n_omega, n_kh, n_branch).
    normalized_dispersion_residual: np.ndarray or None
        Optional dimensionless residual for each returned root, with shape
        (n_omega, n_kh, n_branch).
    mode_ambiguous_mask: np.ndarray or None
        Optional boolean MHD classification/root-count ambiguity mask with
        per-cell shape (n_omega, n_kh).
    all_kz_roots: np.ndarray or None
        Optional complete complex MHD root set in km^-1, with shape
        (n_omega, n_kh, n_all_roots). Complete-root storage can contain more
        branch slots than the returned slow- or fast-mode pair.
    mode_classification: np.ndarray or None
        Optional integer classifications in complete solver-root order, with
        shape (n_omega, n_kh, n_all_roots): -1 for slow, 0 for unresolved,
        and +1 for fast.
    branch_tracking_confidence: np.ndarray or None
        Optional per-cell assignment confidence, with shape (n_omega, n_kh).
    selected_root_indices: np.ndarray or None
        Optional indices into the complete solver-root set, with returned-root
        shape (n_omega, n_kh, n_branch); -1 denotes a missing selected root.

    Outputs
    -------
    result: DiagramResult
        Frozen result container. Array contents remain mutable by convention;
        callers treat cached results as read-only.
    '''

    # Store the model state and display axes for this calculation
    model_key: str
    parameter_values: tuple[tuple[str, float], ...]
    kh_mm_inv: np.ndarray
    nu_mhz: np.ndarray
    # Store the branch-indexed fields and per-cell numerical masks
    phase_deg: np.ndarray
    kz_real: np.ndarray
    kz_imag: np.ndarray
    numerical_warning_mask: np.ndarray
    branch_identity_uncertain_mask: np.ndarray
    # Store complete and branch-associated reference curves in display units
    omega_boundaries_mhz: np.ndarray
    associated_boundaries_mhz: np.ndarray
    fmode_nu_mhz: np.ndarray
    # Store display names, numerical summaries, warnings, and elapsed time
    branch_labels: tuple[str, ...]
    boundary_labels: tuple[str, ...]
    summary: Mapping[str, float | int | str | bool]
    warnings: tuple[str, ...]
    elapsed_seconds: float
    # Retain optional MHD propagation, mode-selection, and confidence diagnostics
    propagating_mask: np.ndarray | None = None
    vertical_group_velocity_km_s: np.ndarray | None = None
    normalized_dispersion_residual: np.ndarray | None = None
    mode_ambiguous_mask: np.ndarray | None = None
    all_kz_roots: np.ndarray | None = None
    mode_classification: np.ndarray | None = None
    branch_tracking_confidence: np.ndarray | None = None
    selected_root_indices: np.ndarray | None = None


class InvalidParameterError(ValueError):
    '''
    Purpose
    -------
    Raised before a solver call when an input is outside its safe range.

    Inputs
    ------
    message: str
        Explanation of a rejected model parameter or calculation-grid input.

    Outputs
    -------
    exception: InvalidParameterError
        ValueError subclass raised before or during parameter normalization.
    '''


@dataclass(frozen=True)
class BranchAlignment:
    '''
    Purpose
    -------
    Diagnostics for one global branch-axis assignment between calculations.

    Inputs
    ------
    permutation: tuple of int
        Current-result branch indices in reference-result panel order.
    total_cost, second_best_cost: float
        Best chosen global assignment cost and next sorted candidate cost.
    ambiguous: bool
        Whether multiple assignments lie within the tie tolerance.
    reliable_sample_fraction, valid_sample_fraction: float
        Fraction of fully reliable samples and minimum valid coverage among
        the chosen branch pairs, respectively.

    Outputs
    -------
    alignment: BranchAlignment
        Frozen diagnostics describing one global branch permutation.
    '''

    # Store the global assignment and its cost, ambiguity, and coverage diagnostics
    permutation: tuple[int, ...]
    total_cost: float
    second_best_cost: float
    ambiguous: bool
    reliable_sample_fraction: float
    valid_sample_fraction: float


def _p(
    key: str,
    label: str,
    unit: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    role: str = "dispersion",
    help_text: str = "",
) -> ParameterDefinition:
    '''
    Purpose
    -------
    Create a parameter definition with normalized numeric values.

    Inputs
    ------
    key, label, unit: str
        Parameter identifier, displayed name, and physical unit.
    default, minimum, maximum, step: float
        Initial value and suggested slider interval/increment.
    role: str, optional
        Change category, normally 'dispersion' or 'diagram'.
    help_text: str, optional
        Supplemental explanatory text.

    Outputs
    -------
    definition: ParameterDefinition
        Frozen metadata with all numeric control settings converted to float.
    '''

    # Normalize control settings before constructing the frozen metadata record
    return ParameterDefinition(
        key=key,
        label=label,
        unit=unit,
        default=float(default),
        minimum=float(minimum),
        maximum=float(maximum),
        step=float(step),
        role=role,
        help_text=help_text,
    )


def _angle_defaults(model_key: str) -> tuple[float, float]:
    '''
    Purpose
    -------
    Return the configured inclination and azimuth in degrees.

    Inputs
    ------
    model_key: str
        Key of a configured model containing theta and phi in radians.

    Outputs
    -------
    theta_deg, phi_deg: tuple of float
        Configured field inclination and azimuth converted to degrees.
    '''

    # Convert configured radian angles into the units displayed by the controls
    params = config.params[model_key]
    return float(np.degrees(params["theta"])), float(np.degrees(params["phi"]))


# Read model-specific angular defaults before constructing the control registry
_nc_theta, _nc_phi = _angle_defaults("nc2009")
_alfven_theta, _alfven_phi = _angle_defaults("alfven_mode")
_sf_theta, _sf_phi = _angle_defaults("slow_fast_mode")
_slow_theta, _slow_phi = _angle_defaults("slow_mode")
_fast_theta, _fast_phi = _angle_defaults("fast_mode")


# Define the visible models and controls in their tab display order
MODEL_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition(
        # Expose independent speed, buoyancy, cutoff, and magnetic-geometry values
        key="nc2009",
        title="3D MAG Wave k–ω Diagram",
        reference="Newington & Cally (2010)",
        parameters=(
            _p("a", "Alfvén speed a", "km s⁻¹", config.params["nc2009"]["a"], 0.0, 40.0, 0.05),
            _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["nc2009"]["cs"], 1.0, 30.0, 0.1),
            _p("N2", "Squared Brunt–Väisälä frequency N²", "s⁻²", config.params["nc2009"]["N"]**2, -0.0225, 0.0225, 0.00001),
            _p("wac", "Acoustic-cutoff angular frequency ωac", "rad s⁻¹", config.params["nc2009"]["wac"], 0.001, 0.15, 0.001),
            _p("theta_deg", "Field inclination θ", "deg", _nc_theta, -180.0, 180.0, 1.0),
            _p("phi_deg", "Field azimuth φ", "deg", _nc_phi, -180.0, 180.0, 1.0),
            _p("dz", "Height separation Δz", "km", config.params["nc2009"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
            _p("g", "Gravity g (f-mode reference)", "km s⁻²", config.params["nc2009"]["g"], 0.01, 1.0, 0.001, role="diagram"),
        ),
    ),
    ModelDefinition(
        # Expose the homogeneous Alfvén relation and its diagram-only references
        key="alfven_mode",
        title="Homogeneous-MHD Alfvén k–ω Diagram",
        reference="Exact analytical homogeneous ideal-MHD Alfvén branch",
        parameters=(
            _p("a", "Alfvén speed a", "km s⁻¹", config.params["alfven_mode"]["a"], 0.0, 40.0, 0.05),
            _p("theta_deg", "Field inclination θ", "deg", _alfven_theta, -180.0, 180.0, 1.0),
            _p("phi_deg", "Field azimuth φ", "deg", _alfven_phi, -180.0, 180.0, 1.0),
            _p("dz", "Height separation Δz", "km", config.params["alfven_mode"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
            _p("g", "Gravity g (f-mode reference)", "km s⁻²", config.params["alfven_mode"]["g"], 0.01, 1.0, 0.001, role="diagram"),
        ),
        metadata_note="Exact homogeneous ideal-MHD relation; excludes stratification and mode conversion. Horizontal and zero fields can make k_z non-isolated; no denominator clipping is applied.",
    ),
    ModelDefinition(
        # Expose the separately classified slow magnetoacoustic family
        key="slow_mode",
        title="Exact Slow Magnetoacoustic k–ω Diagram",
        reference="Homogeneous ideal-MHD slow branch",
        parameters=(
            _p("a", "Alfvén speed a", "km s⁻¹", config.params["slow_mode"]["a"], 0.0, 40.0, 0.05),
            _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["slow_mode"]["cs"], 1.0, 30.0, 0.1),
            _p("theta_deg", "Field inclination θ", "deg", _slow_theta, -180.0, 180.0, 1.0),
            _p("phi_deg", "Field azimuth φ", "deg", _slow_phi, -180.0, 180.0, 1.0),
            _p("dz", "Height separation Δz", "km", config.params["slow_mode"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
            _p("g", "Gravity g (f-mode reference)", "km s⁻²", config.params["slow_mode"]["g"], 0.01, 1.0, 0.001, role="diagram"),
        ),
        metadata_note="Exact homogeneous ideal-MHD dispersion; excludes stratification and mode conversion.",
    ),
    ModelDefinition(
        # Expose the separately classified fast magnetoacoustic family
        key="fast_mode",
        title="Exact Fast Magnetoacoustic k–ω Diagram",
        reference="Homogeneous ideal-MHD fast branch",
        parameters=(
            _p("a", "Alfvén speed a", "km s⁻¹", config.params["fast_mode"]["a"], 0.0, 40.0, 0.05),
            _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["fast_mode"]["cs"], 1.0, 30.0, 0.1),
            _p("theta_deg", "Field inclination θ", "deg", _fast_theta, -180.0, 180.0, 1.0),
            _p("phi_deg", "Field azimuth φ", "deg", _fast_phi, -180.0, 180.0, 1.0),
            _p("dz", "Height separation Δz", "km", config.params["fast_mode"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
            _p("g", "Gravity g (f-mode reference)", "km s⁻²", config.params["fast_mode"]["g"], 0.01, 1.0, 0.001, role="diagram"),
        ),
        metadata_note="Exact homogeneous ideal-MHD dispersion; excludes stratification and mode conversion.",
    ),
    ModelDefinition(
        # Expose active rational-dispersion inputs and retain inactive metadata in a
        # note
        key="bunte1993",
        title="Bunte 1993 k–ω Diagram",
        reference="Bunte & Bogdan (1993)",
        parameters=(
            _p("a", "Alfvén speed a", "km s⁻¹", config.params["bunte1993"]["a"], 0.0, 40.0, 0.05),
            _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["bunte1993"]["cs"], 1.0, 30.0, 0.1),
            _p("g", "Gravity g", "km s⁻²", config.params["bunte1993"]["g"], 0.01, 1.0, 0.001),
            _p("gamma", "Adiabatic index γ", "dimensionless", config.params["bunte1993"]["gamma"], 1.01, 2.0, 0.01),
            _p("H", "Density scale height H", "km", config.params["bunte1993"]["H"], 20.0, 1000.0, 5.0),
            _p("tau", "Radiative damping time τ", "s", config.params["bunte1993"]["tau"], 0.1, 1000.0, 1.0),
            _p("dz", "Height separation Δz", "km", config.params["bunte1993"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
        ),
        metadata_note=(
            "The current solver stores ε and the configuration stores B, but neither "
            "appears in the implemented polynomial coefficients. They are therefore "
            "reported as inactive metadata, not offered as misleading controls."
        ),
    ),
    ModelDefinition(
        # Expose the radiative hydrodynamic parameters used by the existing solver
        key="mt1982",
        title="MT1982 k–ω Diagram",
        reference="Mihalas & Toomre (1982)",
        parameters=(
            _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["mt1982"]["cs"], 1.0, 30.0, 0.1),
            _p("g", "Gravity g", "km s⁻²", config.params["mt1982"]["g"], 0.01, 1.0, 0.001),
            _p("gamma", "Adiabatic index γ", "dimensionless", config.params["mt1982"]["gamma"], 1.01, 2.0, 0.01),
            _p("H", "Density scale height H", "km", config.params["mt1982"]["H"], 20.0, 1000.0, 5.0),
            _p("N2", "Squared Brunt–Väisälä frequency N²", "s⁻²", config.params["mt1982"]["N"]**2, -0.0225, 0.0225, 0.00001),
            _p("tau", "Radiative damping time τ", "s", config.params["mt1982"]["tau"], 0.1, 1000.0, 1.0),
            _p("dz", "Height separation Δz", "km", config.params["mt1982"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
        ),
        metadata_note=(
            "In the existing implementation the degree-eight ω polynomial uses g "
            "but not N², while the k_z expression uses N² but not g. Its default "
            "candidate ω roots do not verify as k_z=0 boundaries, so they are "
            "diagnosed but intentionally not overplotted as valid boundaries."
        ),
    ),
    ModelDefinition(
        # Expose the adiabatic acoustic-gravity parameters and f-mode reference
        key="mt1981",
        title="MT1981 k–ω Diagram",
        reference="Mihalas & Toomre (1981)",
        parameters=(
            _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["mt1981"]["cs"], 1.0, 30.0, 0.1),
            _p("H", "Density scale height H", "km", config.params["mt1981"]["H"], 20.0, 1000.0, 5.0),
            _p("N2", "Squared Brunt–Väisälä frequency N²", "s⁻²", config.params["mt1981"]["N"]**2, -0.0225, 0.0225, 0.00001),
            _p("dz", "Height separation Δz", "km", config.params["mt1981"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
            _p("g", "Gravity g (f-mode reference)", "km s⁻²", config.params["mt1981"]["g"], 0.01, 1.0, 0.001, role="diagram"),
        ),
    ),
)

# Keep the combined slow/fast model available for saved-state compatibility
_LEGACY_SLOW_FAST_DEFINITION = ModelDefinition(
    key="slow_fast_mode",
    title="Legacy combined slow/fast magnetoacoustic model",
    reference="Compatibility alias for previously saved configurations",
    parameters=(
        _p("a", "Alfvén speed a", "km s⁻¹", config.params["slow_fast_mode"]["a"], 0.0, 40.0, 0.05),
        _p("cs", "Sound speed cₛ", "km s⁻¹", config.params["slow_fast_mode"]["cs"], 1.0, 30.0, 0.1),
        _p("theta_deg", "Field inclination θ", "deg", _sf_theta, -180.0, 180.0, 1.0),
        _p("phi_deg", "Field azimuth φ", "deg", _sf_phi, -180.0, 180.0, 1.0),
        _p("dz", "Height separation Δz", "km", config.params["slow_fast_mode"]["dz"], 1.0, 2000.0, 5.0, role="diagram"),
        _p("g", "Gravity g (f-mode reference)", "km s⁻²", config.params["slow_fast_mode"]["g"], 0.01, 1.0, 0.001, role="diagram"),
    ),
)
# Provide immutable lookup metadata without adding the legacy alias to visible tabs
MODEL_BY_KEY = MappingProxyType({
    **{item.key: item for item in MODEL_DEFINITIONS},
    "slow_fast_mode": _LEGACY_SLOW_FAST_DEFINITION,
})
MODEL_ORDER = tuple(item.key for item in MODEL_DEFINITIONS)


def normalized_parameter_values(
    model_key: str, values: Mapping[str, float] | None = None
) -> tuple[tuple[str, float], ...]:
    '''
    Purpose
    -------
    Merge defaults, reject unknown values, and validate physical parameter domains.

    Inputs
    ------
    model_key: str
        Key in MODEL_BY_KEY, including the legacy slow_fast_mode alias.
    values: Mapping[str, float] or None, optional
        Parameter overrides in displayed units; missing entries use defaults.
        Legacy nonnegative angular N is squared when the model exposes N2.

    Outputs
    -------
    parameter_values: tuple of (str, float)
        Validated values in the model's declared control order, suitable as a
        cache key. Unknown keys, nonfinite values, and invalid physical domains
        raise InvalidParameterError; unknown models raise KeyError.
    '''

    # Resolve the model and copy supplied values before any normalization
    if model_key not in MODEL_BY_KEY:
        raise KeyError(f"Unknown model {model_key!r}.")
    definition = MODEL_BY_KEY[model_key]
    supplied = {} if values is None else dict(values)
    # Backward-compatible saved inputs use angular N, never a signed square.
    if "N" in supplied and "N2" in {item.key for item in definition.parameters}:
        legacy = supplied.pop("N")
        if not np.isfinite(legacy) or legacy < 0:
            raise InvalidParameterError("Legacy N must be finite and nonnegative; use signed N2.")
        if "N2" not in supplied:
            supplied["N2"] = float(legacy)**2
    # Reject unsupported keys after translating any legacy buoyancy input
    allowed = {item.key for item in definition.parameters}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise InvalidParameterError(
            f"Unsupported parameter(s) for {model_key}: {', '.join(unknown)}"
        )

    # Validate physical domains in declared control order for a stable cache key
    normalized: list[tuple[str, float]] = []
    for item in definition.parameters:
        value = float(supplied.get(item.key, item.default))
        if not np.isfinite(value):
            raise InvalidParameterError(f"{item.title} must be finite.")
        try:
            validate_parameter(item.key, value)
        except ValueError as exc:
            raise InvalidParameterError(f"{item.title}: {exc}") from exc
        normalized.append((item.key, value))
    return tuple(normalized)


def build_model_parameters(
    model_key: str, values: Mapping[str, float] | None = None
) -> dict[str, float | int | str]:
    '''
    Purpose
    -------
    Return a fresh legacy parameter dictionary without mutating config.

    Inputs
    ------
    model_key: str
        Configuration/model-definition key.
    values: Mapping[str, float] or None, optional
        Display-unit parameter overrides passed through normalization.

    Outputs
    -------
    params: dict
        Fresh solver dictionary with angles in radians, signed N2 replacing
        legacy N when exposed, and NC2009 field components in km/s. The shared
        configuration dictionaries are not modified.
    '''

    # Copy configured solver parameters and apply validated display-unit overrides
    normalized = dict(normalized_parameter_values(model_key, values))
    params = config.params[model_key].copy()
    theta_deg = normalized.pop("theta_deg", None)
    phi_deg = normalized.pop("phi_deg", None)
    params.update(normalized)
    # Remove the legacy buoyancy value and convert displayed angles to radians
    if "N2" in normalized:
        params.pop("N", None)
    if theta_deg is not None:
        params["theta"] = np.deg2rad(theta_deg)
    if phi_deg is not None:
        params["phi"] = np.deg2rad(phi_deg)

    # NC2009 consumes both the angles and these components.  Recompute them
    # exactly as config.py does whenever a, theta, or phi changes.
    if model_key == "nc2009":
        params["ax"], params["ay"], params["az"] = components(
            params["a"], params["theta"], params["phi"], degrees=False)
    return params


def _finite_max(values: np.ndarray) -> float:
    '''
    Purpose
    -------
    Return the largest finite value, or NaN when no finite value exists.

    Inputs
    ------
    values: array-like
        Numeric values of any shape, converted to a floating-point array.

    Outputs
    -------
    maximum: float
        Largest finite value, or NaN for an empty/all-nonfinite input.
    '''

    # Exclude NaN and infinity before reporting the largest available diagnostic
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def _branch_labels(
    kz_real: np.ndarray,
    model_key: str | None = None,
    group_velocity_z: np.ndarray | None = None,
) -> tuple[str, ...]:
    '''
    Purpose
    -------
    Describe branches by solver index or their vertical energy direction.

    Inputs
    ------
    kz_real: np.ndarray
        Real roots with shape (n_omega, n_kh, n_branch), in km^-1.
    model_key: str or None, optional
        Model identifier used to select physical MHD branch names.
    group_velocity_z: np.ndarray or None, optional
        Root-wise vertical group velocities (km/s), matching kz_real's shape.

    Outputs
    -------
    labels: tuple of str
        One name per branch. MHD energy-direction labels use finite group
        velocities; other models report tracked indices and median root signs.
    '''

    # Build one descriptive label for each returned branch slot
    labels: list[str] = []
    for branch in range(kz_real.shape[2]):
        # Use group-velocity signs to describe energy direction for classified MHD modes
        if model_key in {"alfven_mode", "slow_mode", "fast_mode"}:
            prefix = {
                "alfven_mode": "Alfvén",
                "slow_mode": "Slow magnetoacoustic",
                "fast_mode": "Fast magnetoacoustic",
            }[model_key]
            velocities = (
                np.asarray(group_velocity_z)[:, :, branch]
                if group_velocity_z is not None
                else np.asarray([])
            )
            velocities = velocities[np.isfinite(velocities)]
            if velocities.size and np.all(velocities > 0.0):
                labels.append(f"{prefix}; upward energy propagation")
            elif velocities.size and np.all(velocities < 0.0):
                labels.append(f"{prefix}; downward energy propagation")
            elif velocities.size:
                labels.append(f"{prefix}; vertical energy direction varies")
            else:
                labels.append(f"{prefix}; energy direction undefined")
            continue
        # Summarize other tracked branches by the sign of their finite median root
        values = kz_real[:, :, branch]
        finite = values[np.isfinite(values)]
        suffix = ""
        if finite.size:
            median = float(np.median(finite))
            if median > 1e-12:
                suffix = "; median Re(k_z) > 0"
            elif median < -1e-12:
                suffix = "; median Re(k_z) < 0"
        labels.append(f"Tracked k_z branch {branch + 1}{suffix}")
    return tuple(labels)


def _boundary_labels(model_key: str, boundaries: np.ndarray) -> tuple[str, ...]:
    '''
    Purpose
    -------
    Label dispersion boundaries, identifying slow and fast branches where possible.

    Inputs
    ------
    model_key: str
        Model identifier selecting generic or magnetoacoustic labels.
    boundaries: np.ndarray
        Frequencies with shape (n_kh, n_boundary); only relative frequency
        ordering is used for the legacy combined slow/fast labels.

    Outputs
    -------
    labels: tuple of str
        Boundary names; legacy slow/fast labels use the lowest/highest finite
        median frequencies, retaining generic names for other boundary slots.
    '''

    # Start with generic names and handle the separately selected MHD modes
    count = boundaries.shape[1]
    labels = [f"Dispersion boundary {index + 1}" for index in range(count)]
    if model_key == "slow_mode":
        return ("Slow magnetoacoustic boundary",)
    if model_key == "fast_mode":
        return ("Fast magnetoacoustic boundary",)
    if model_key != "slow_fast_mode" or count < 2:
        return tuple(labels)

    # Identify the lowest and highest finite curves for the legacy combined model
    medians = np.full(count, np.inf)
    for index in range(count):
        finite = boundaries[:, index][np.isfinite(boundaries[:, index])]
        if finite.size:
            medians[index] = float(np.median(finite))
    finite_order = [index for index in np.argsort(medians) if np.isfinite(medians[index])]
    if finite_order:
        labels[finite_order[0]] = "Slow magnetoacoustic branch"
    if len(finite_order) > 1:
        labels[finite_order[-1]] = "Fast magnetoacoustic branch"
    return tuple(labels)


def _compute_diagram_uncached(
    model_key: str,
    parameter_values: tuple[tuple[str, float], ...],
    n_kh: int,
    n_omega: int,
    axes: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> DiagramResult:
    '''
    Purpose
    -------
    Run the selected dispersion solver and assemble its interactive result.

    Inputs
    ------
    model_key: str
        Model to solve using config and the exposed parameter definitions.
    parameter_values: tuple of (str, float)
        Normalized display-unit parameter values.
    n_kh, n_omega: int
        Grid dimensions; preset dimensions must each be between 8 and 220.
    axes: tuple of tuple of float or None, optional
        Explicit kh (Mm^-1) and frequency (mHz) coordinates; bypasses preset
        dimension limits and preserves supplied bins.

    Outputs
    -------
    result: DiagramResult
        Roots, phase differences, associated boundaries, masks, labels, and
        numerical diagnostics on the requested grid. Boundary-association
        failures become warnings/NaN curves; invalid grids or absent roots
        raise exceptions except for supported non-isolated MHD limits.
    '''

    # Enforce demonstration-grid limits only when explicit axes were not supplied
    if axes is None and (n_kh < 8 or n_omega < 8):
        raise InvalidParameterError("Each grid dimension must contain at least 8 samples.")
    if axes is None and (n_kh > 220 or n_omega > 220):
        raise InvalidParameterError("Calculation grids are limited to 220 × 220 samples.")

    # Start timing and construct an independent solver parameter dictionary
    started = perf_counter()
    params = build_model_parameters(model_key, dict(parameter_values))
    # Preserve the notebook's operation order exactly. Algebraically equivalent
    # divide-by-conversion expressions can differ in the last bit and amplify
    # near a degenerate NC2009 root pair.
    kh = (np.asarray(axes[0]) if axes is not None else
          np.linspace(DEFAULT_DOMAIN[0], DEFAULT_DOMAIN[1], n_kh)) * 1e-3
    if axes is not None:
        # Preserve native bins, including singular zero-frequency cells. Their
        # existing numerical-warning mask remains authoritative.
        omega = np.asarray(axes[1]) * 2 * pi * 1e-3
    elif model_key == "bunte1993":
        # This rational wave relation has a pole at omega=0. Sample positive
        # image-cell centers inside the unchanged 0..11 mHz plotting domain.
        edges = np.linspace(DEFAULT_DOMAIN[2], DEFAULT_DOMAIN[3], n_omega + 1)
        omega = (edges[:-1] + edges[1:]) * 0.5 * 2 * pi * 1e-3
    else:
        omega = np.linspace(DEFAULT_DOMAIN[2], DEFAULT_DOMAIN[3], n_omega) * 2 * pi * 1e-3
    # Build the physical grid used by the dispersion-relation solver
    kh_grid, omega_grid = np.meshgrid(kh, omega)
    diagram = DiagnosticDiagram(kh_grid, omega_grid, params)

    # Solve frequency boundaries and vertical roots while retaining numerical
    # diagnostics
    association_error = ""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        omega_boundaries, omega_diagnostics = diagram.omega_solve_grid(
            return_diagnostics=True
        )
        # Use the direct quadratic root path for the Bunte relation
        if model_key == "bunte1993":
            kz, kz_diagnostics = diagram.kz_quadratic_roots_grid(return_diagnostics=True)
        else:
            kz, kz_diagnostics = diagram.kz_solve_grid(return_diagnostics=True)
        # Associate boundary frequencies with the globally tracked vertical-root
        # families
        try:
            associated, association_diagnostics = (
                diagram.associate_omega_boundaries_with_kz(
                    omega_boundaries, kz, return_diagnostics=True
                )
            )
        except (ArithmeticError, FloatingPointError, ValueError) as exc:
            # Retain a missing-curve result when boundary association cannot be
            # completed
            associated = np.full(
                (n_kh, int(params["kz_order"]), omega_boundaries.shape[1]),
                np.nan,
            )
            association_diagnostics = {}
            association_error = str(exc)

        # Calculate phase difference separately for every returned vertical-root branch
        phase_deg = np.full(kz.shape, np.nan, dtype=float)
        for branch in range(kz.shape[2]):
            phase_speed = diagram.phase_speed(omega_grid, kz[:, :, branch])
            phase = diagram.phase_difference(omega_grid, phase_speed, params["dz"])
            phase_deg[:, :, branch] = np.real(phase) * 180.0 / pi
        # The notebook's nan_to_num plotting path displays the indeterminate
        # omega=0 limit as zero phase. Preserve that convention only on this
        # known endpoint; nonfinite solver output elsewhere remains NaN.
        phase_deg[np.isclose(omega_grid, 0.0, rtol=0.0, atol=0.0), :] = 0.0

    # Separate complex root components and collect masks used by the interactive plots
    kz_real = np.real(kz).astype(float, copy=False)
    kz_imag = np.imag(kz).astype(float, copy=False)
    warning_mask = np.asarray(
        kz_diagnostics.get("plot_suspicious_mask", np.zeros(kz.shape[:2], dtype=bool)),
        dtype=bool,
    )
    mode_ambiguous_mask = np.asarray(
        kz_diagnostics.get("mode_ambiguous_mask", np.zeros(kz.shape[:2], dtype=bool)),
        dtype=bool,
    )
    if model_key in {"alfven_mode", "slow_mode", "fast_mode"}:
        warning_mask |= mode_ambiguous_mask
    # Combine available failure, degeneracy, and identity masks on the image grid
    branch_identity_uncertain_mask = np.zeros(kz.shape[:2], dtype=bool)
    for diagnostic_name in (
        "solve_failed_mask",
        "solver_instability_mask",
        "plot_suspicious_mask",
        "uncertain_mask",
        "branch_degenerate_mask",
        "mode_ambiguous_mask",
    ):
        diagnostic_mask = np.asarray(
            kz_diagnostics.get(diagnostic_name, False), dtype=bool
        )
        if diagnostic_mask.shape == kz.shape[:2]:
            branch_identity_uncertain_mask |= diagnostic_mask
    # Reject entirely unavailable roots except for supported non-isolated MHD limits
    finite_kz = np.isfinite(kz.real) & np.isfinite(kz.imag)
    if not np.any(finite_kz) and model_key not in {"alfven_mode", "slow_mode"}:
        raise RuntimeError("The selected parameters produced no finite k_z roots.")

    # Summarize solver failures, mode ambiguity, and verified boundary coverage
    solve_failed_count = int(
        np.count_nonzero(kz_diagnostics.get("solve_failed_mask", False))
    )
    unstable_count = int(
        np.count_nonzero(kz_diagnostics.get("solver_instability_mask", False))
    )
    suspicious_count = int(np.count_nonzero(warning_mask))
    tracking_uncertain_count = int(
        np.count_nonzero(kz_diagnostics.get("uncertain_mask", False))
    )
    mode_ambiguous_count = int(np.count_nonzero(mode_ambiguous_mask))
    unmatched_boundary_count = int(
        np.count_nonzero(association_diagnostics.get("unmatched_boundary_mask", False))
    )
    associated_boundary_count = int(np.count_nonzero(np.isfinite(associated)))
    finite_kz_fraction = float(np.count_nonzero(finite_kz) / finite_kz.size)

    # Translate nonzero quality counts into visible calculation warnings
    result_warnings: list[str] = []
    if solve_failed_count:
        result_warnings.append(f"{solve_failed_count} grid cells had failed polynomial solves")
    if unstable_count:
        result_warnings.append(f"{unstable_count} grid cells exceeded the root-residual tolerance")
    if suspicious_count:
        result_warnings.append(f"{suspicious_count} grid cells are masked as numerical warnings")
    if finite_kz_fraction < 1.0:
        result_warnings.append(
            f"finite k_z coverage is {100.0 * finite_kz_fraction:.1f}% of root slots"
        )
    if not np.any(finite_kz):
        result_warnings.append(
            "No isolated Alfvén k_z roots exist for this geometry/frequency grid"
            if model_key == "alfven_mode"
            else "No isolated slow-mode k_z roots exist for this zero-field frequency grid"
        )
    if tracking_uncertain_count:
        result_warnings.append(
            f"{tracking_uncertain_count} cells have ambiguous branch identity near "
            "a degeneracy or crossing (the roots themselves remain available)"
        )
    if mode_ambiguous_count:
        result_warnings.append(
            f"{mode_ambiguous_count} cells have ambiguous MHD mode identity or root count"
        )
    if unmatched_boundary_count:
        result_warnings.append(
            f"{unmatched_boundary_count} candidate ω roots were not verified as k_z=0 boundaries"
        )
    if association_error:
        result_warnings.append(f"boundary association failed: {association_error}")

    # Collect compact numerical diagnostics without exposing mutable summary state
    summary: dict[str, float | int | str | bool] = {
        "grid_source": "explicit_axes" if axes is not None else "preset",
        "n_kh": int(n_kh),
        "n_omega": int(n_omega),
        "kz_branch_count": int(kz.shape[2]),
        "omega_boundary_count": int(omega_boundaries.shape[1]),
        "finite_kz_fraction": finite_kz_fraction,
        "kz_solve_failed_cells": solve_failed_count,
        "kz_unstable_cells": unstable_count,
        "kz_warning_cells": suspicious_count,
        "kz_tracking_uncertain_cells": tracking_uncertain_count,
        "kz_mode_ambiguous_cells": mode_ambiguous_count,
        "omega_solve_failed_columns": int(
            np.count_nonzero(omega_diagnostics.get("solve_failed_mask", False))
        ),
        "unmatched_boundary_points": unmatched_boundary_count,
        "associated_boundary_points": associated_boundary_count,
        "max_kz_root_residual": _finite_max(
            kz_diagnostics.get("max_root_residual", np.asarray([]))
        ),
        "max_omega_root_residual": _finite_max(
            omega_diagnostics.get("max_root_residual", np.asarray([]))
        ),
        "association_error": association_error,
    }

    # Package display-unit fields and retain optional model-specific solver diagnostics
    return DiagramResult(
        model_key=model_key,
        parameter_values=parameter_values,
        kh_mm_inv=np.asarray(axes[0]) if axes is not None else kh * INV_KM_TO_INV_MM,
        nu_mhz=np.asarray(axes[1]) if axes is not None else omega * RAD_S_TO_MHZ,
        phase_deg=phase_deg,
        kz_real=kz_real,
        kz_imag=kz_imag,
        numerical_warning_mask=warning_mask,
        branch_identity_uncertain_mask=branch_identity_uncertain_mask,
        omega_boundaries_mhz=omega_boundaries * RAD_S_TO_MHZ,
        associated_boundaries_mhz=associated * RAD_S_TO_MHZ,
        fmode_nu_mhz=diagram.fmode_dispersion() * RAD_S_TO_MHZ,
        branch_labels=_branch_labels(
            kz_real,
            model_key,
            kz_diagnostics.get(
                "selected_vertical_group_velocity_km_s",
                kz_diagnostics.get("vertical_group_velocity_km_s"),
            ),
        ),
        boundary_labels=_boundary_labels(model_key, omega_boundaries),
        summary=MappingProxyType(summary),
        warnings=tuple(result_warnings),
        elapsed_seconds=perf_counter() - started,
        propagating_mask=kz_diagnostics.get(
            "selected_propagating_root_mask", kz_diagnostics.get("propagating_root_mask")
        ),
        vertical_group_velocity_km_s=kz_diagnostics.get(
            "selected_vertical_group_velocity_km_s",
            kz_diagnostics.get("vertical_group_velocity_km_s"),
        ),
        normalized_dispersion_residual=kz_diagnostics.get(
            "selected_normalized_dispersion_residual",
            kz_diagnostics.get(
                "normalized_magnetoacoustic_residual",
                kz_diagnostics.get("normalized_alfven_residual"),
            ),
        ),
        mode_ambiguous_mask=(
            kz_diagnostics.get("mode_ambiguous_mask")
            if model_key in {"alfven_mode", "slow_mode", "fast_mode"}
            else None
        ),
        all_kz_roots=kz_diagnostics.get(
            "tracked_all_roots", kz if model_key in {"alfven_mode", "slow_mode", "fast_mode"} else None
        ),
        mode_classification=kz_diagnostics.get("mode_classification"),
        branch_tracking_confidence=kz_diagnostics.get("assignment_confidence"),
        selected_root_indices=kz_diagnostics.get("selected_root_indices"),
    )


@lru_cache(maxsize=24)
def _cached_compute_diagram(
    model_key: str,
    parameter_values: tuple[tuple[str, float], ...],
    n_kh: int,
    n_omega: int,
    axes: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> DiagramResult:
    '''
    Purpose
    -------
    Return a cached diagram result or calculate and cache a new one.

    Inputs
    ------
    model_key: str
        Model identifier.
    parameter_values: tuple of (str, float)
        Hashable normalized parameter values in display units.
    n_kh, n_omega: int
        Calculation dimensions.
    axes: tuple of tuple of float or None, optional
        Exact explicit axes in Mm^-1 and mHz, included in the cache key.

    Outputs
    -------
    result: DiagramResult
        Cached or newly computed result. The LRU cache retains at most 24
        argument combinations and returns shared result objects.
    '''

    # Populate the LRU cache on a miss using the complete normalized state
    return _compute_diagram_uncached(model_key, parameter_values, n_kh, n_omega, axes)


def compute_diagram(
    model_key: str,
    values: Mapping[str, float] | None = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    *,
    use_cache: bool = True,
    kh_mm_inv=None,
    nu_mhz=None,
) -> DiagramResult:
    '''
    Purpose
    -------
    Solve on a preset, or on both supplied comparison axes without resampling.

    Explicit axes override ``resolution`` and its demonstration-size limits.
    Their exact coordinates are included in the result cache key.

    Inputs
    ------
    model_key: str
        Key of the model to solve.
    values: Mapping[str, float] or None, optional
        Display-unit parameter overrides; missing values use model defaults.
    resolution: tuple of int, optional
        Preset (n_kh, n_omega) dimensions, each between 8 and 220.
    use_cache: bool, optional
        Reuse the shared 24-entry LRU result cache when True.
    kh_mm_inv, nu_mhz: array-like or None, optional
        Both explicit one-dimensional axes, in Mm^-1 and mHz. Each must contain
        at least two finite, strictly increasing, nonnegative samples.

    Outputs
    -------
    result: DiagramResult
        Diagram on the chosen axes, with unavailable solver values retained as
        NaN and calculation warnings recorded in the result. Invalid parameters
        or axes raise InvalidParameterError.
    '''

    # Normalize the physical parameters before validating or caching the grid
    parameter_values = normalized_parameter_values(model_key, values)
    axes = None
    # Require both explicit axes and preserve their exact increasing coordinates
    if kh_mm_inv is not None or nu_mhz is not None:
        if kh_mm_inv is None or nu_mhz is None:
            raise InvalidParameterError("Both comparison axes are required.")
        normalized_axes = []
        for label, values_axis in (("k_h", kh_mm_inv), ("frequency", nu_mhz)):
            axis = np.asarray(values_axis, dtype=float)
            if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
                    or axis[0] < 0 or np.any(np.diff(axis) <= 0)):
                raise InvalidParameterError(
                    f"The {label} axis must contain at least two finite, increasing, nonnegative samples.")
            normalized_axes.append(tuple(map(float, axis)))
        axes = tuple(normalized_axes)
        n_kh, n_omega = map(len, axes)
    else:
        n_kh, n_omega = (int(resolution[0]), int(resolution[1]))
    # Select the shared cache or force a fresh calculation with the same inputs
    if use_cache:
        return _cached_compute_diagram(model_key, parameter_values, n_kh, n_omega, axes)
    return _compute_diagram_uncached(model_key, parameter_values, n_kh, n_omega, axes)


def update_diagram_postprocessing(
    result: DiagramResult, values: Mapping[str, float]
) -> DiagramResult:
    '''
    Purpose
    -------
    Update Δz/f-mode-only values without repeating any polynomial solve.

    Inputs
    ------
    result: DiagramResult
        Previously solved result whose roots and grid are reused.
    values: Mapping[str, float]
        Desired complete parameter state in display units; omitted keys fall
        back to model defaults through normalization.

    Outputs
    -------
    updated: DiagramResult
        Replacement result with revised phase, f-mode reference, parameter
        values, summary, and timing. Root arrays are reused. Changes to any
        dispersion-role parameter raise InvalidParameterError.
    '''

    # Compare normalized states before reusing an existing root solution
    started = perf_counter()
    parameter_values = normalized_parameter_values(result.model_key, values)
    previous = dict(result.parameter_values)
    current = dict(parameter_values)
    definition = MODEL_BY_KEY[result.model_key]
    dispersion_keys = {
        item.key for item in definition.parameters if item.role == "dispersion"
    }
    changed_dispersion = [
        key for key in dispersion_keys if current[key] != previous[key]
    ]
    # Prevent a postprocessing update from silently retaining outdated roots
    if changed_dispersion:
        raise InvalidParameterError(
            "Postprocessing-only update cannot change dispersion input(s): "
            + ", ".join(sorted(changed_dispersion))
        )

    # Reconstruct the original grid convention and update only derived diagram fields
    params = build_model_parameters(result.model_key, current)
    explicit_axes = result.summary.get("grid_source") == "explicit_axes"
    kh = (result.kh_mm_inv if explicit_axes else np.linspace(
        DEFAULT_DOMAIN[0], DEFAULT_DOMAIN[1], len(result.kh_mm_inv)
    )) * 1e-3
    if explicit_axes or result.model_key == "bunte1993":
        omega = result.nu_mhz * 2 * pi * 1e-3
    else:
        omega = np.linspace(
            DEFAULT_DOMAIN[2], DEFAULT_DOMAIN[3], len(result.nu_mhz)
        ) * 2 * pi * 1e-3
    kh_grid, omega_grid = np.meshgrid(kh, omega)
    diagram = DiagnosticDiagram(kh_grid, omega_grid, params)
    # Reuse complex roots and recompute the phase for the requested height separation
    kz = result.kz_real + 1j * result.kz_imag
    phase_deg = np.full(kz.shape, np.nan, dtype=float)
    with np.errstate(all="ignore"):
        for branch in range(kz.shape[2]):
            speed = diagram.phase_speed(omega_grid, kz[:, :, branch])
            phase = diagram.phase_difference(omega_grid, speed, params["dz"])
            phase_deg[:, :, branch] = np.real(phase) * 180.0 / pi
        phase_deg[np.isclose(omega_grid, 0.0, rtol=0.0, atol=0.0), :] = 0.0

    # Return a replacement result while sharing the unchanged solver arrays
    summary = dict(result.summary)
    summary["postprocessing_only"] = True
    return replace(
        result,
        parameter_values=parameter_values,
        phase_deg=phase_deg,
        fmode_nu_mhz=diagram.fmode_dispersion() * RAD_S_TO_MHZ,
        summary=MappingProxyType(summary),
        elapsed_seconds=perf_counter() - started,
    )


def _branch_alignment_samples(
    current: DiagramResult, reference: DiagramResult
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Purpose
    -------
    Return corresponding complex-sheet samples and reliability weights.

    Inputs
    ------
    current, reference: DiagramResult
        Calculations whose complete complex branch sheets are compared.

    Outputs
    -------
    current_samples, reference_samples: np.ndarray
        Complex arrays with shape (n_sample, n_branch), in km^-1. Identical
        grids use all cells; differing grids use 18 by 18 normalized positions.
    weights: np.ndarray, float
        Per-sample reliability: 0 for numerical warnings, 0.15 for identity
        uncertainty, and 1 for fully reliable cells.
    '''

    # Reconstruct complete complex branch sheets before comparing corresponding samples
    current_kz = current.kz_real + 1j * current.kz_imag
    reference_kz = reference.kz_real + 1j * reference.kz_imag
    same_grid = (
        current_kz.shape[:2] == reference_kz.shape[:2]
        and np.array_equal(current.kh_mm_inv, reference.kh_mm_inv)
        and np.array_equal(current.nu_mhz, reference.nu_mhz)
    )
    # Use every cell when both results have exactly matching physical axes
    if same_grid:
        current_samples = current_kz.reshape(-1, current_kz.shape[2])
        reference_samples = reference_kz.reshape(-1, reference_kz.shape[2])
        strict_warning = (
            current.numerical_warning_mask | reference.numerical_warning_mask
        ).ravel()
        uncertain = (
            current.branch_identity_uncertain_mask
            | reference.branch_identity_uncertain_mask
        ).ravel()
        weights = np.where(strict_warning, 0.0, np.where(uncertain, 0.15, 1.0))
        return current_samples, reference_samples, weights

    # Preset app resolutions span the same physical domain. Compare differing
    # grids at matching normalized index positions, including when explicit axes
    # have different extents, without interpolating across cuts or NaN regions.
    fractions = np.linspace(0.06, 0.94, 18)
    current_y = np.rint(fractions * (current_kz.shape[0] - 1)).astype(int)
    current_x = np.rint(fractions * (current_kz.shape[1] - 1)).astype(int)
    reference_y = np.rint(fractions * (reference_kz.shape[0] - 1)).astype(int)
    reference_x = np.rint(fractions * (reference_kz.shape[1] - 1)).astype(int)

    # Sample the same normalized positions without interpolating across root-sheet cuts
    current_samples = current_kz[
        current_y[:, None], current_x[None, :], :
    ].reshape(-1, current_kz.shape[2])
    reference_samples = reference_kz[
        reference_y[:, None], reference_x[None, :], :
    ].reshape(-1, reference_kz.shape[2])
    # Assign zero or reduced weight to numerically unreliable sample pairs
    current_uncertain = current.branch_identity_uncertain_mask[
        current_y[:, None], current_x[None, :]
    ]
    reference_uncertain = reference.branch_identity_uncertain_mask[
        reference_y[:, None], reference_x[None, :]
    ]
    strict_warning = (
        current.numerical_warning_mask[current_y[:, None], current_x[None, :]]
        | reference.numerical_warning_mask[
            reference_y[:, None], reference_x[None, :]
        ]
    ).ravel()
    uncertain = (current_uncertain | reference_uncertain).ravel()
    weights = np.where(strict_warning, 0.0, np.where(uncertain, 0.15, 1.0))
    return current_samples, reference_samples, weights


def align_diagram_result_branches(
    current: DiagramResult, reference: DiagramResult
) -> tuple[DiagramResult, BranchAlignment]:
    '''
    Purpose
    -------
    Align a new solve to prior panel identities with one global permutation.

    Compare complete complex ``k_z`` sheets and apply one branch-axis
    permutation to display fields and shape-compatible root diagnostics.
    Complete-root mode classifications and per-cell tracking confidence remain
    unchanged. Pointwise root ordering is preserved through crossings.

    Inputs
    ------
    current, reference: DiagramResult
        Same-model calculations with equal returned branch counts; reference
        supplies the desired panel identities.

    Outputs
    -------
    aligned: DiagramResult
        Replacement current result with one global permutation applied to
        compatible branch-indexed arrays and associated boundary curves.
    diagnostics: BranchAlignment
        Chosen assignment, costs, ambiguity, and sample coverage. Numerically
        tied assignments prefer fewer slot changes. Model or branch-count
        mismatches raise ValueError.
    '''

    # Require compatible model identities and branch counts before assignment
    if current.model_key != reference.model_key:
        raise ValueError("Branch alignment requires results from the same model.")
    branch_count = current.kz_real.shape[2]
    if reference.kz_real.shape[2] != branch_count:
        raise ValueError("Branch alignment requires equal branch counts.")

    # Obtain corresponding complex-root samples and their reliability weights
    current_samples, reference_samples, reliability_weights = _branch_alignment_samples(
        current, reference
    )

    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        '''
        Purpose
        -------
        Return the first sorted value reaching half the total sample weight.

        Inputs
        ------
        values, weights: np.ndarray
            Corresponding nonempty one-dimensional samples and positive weights.

        Outputs
        -------
        median: float
            First stably sorted value whose cumulative weight reaches half the
            total, clamped to the final sample index.
        '''

        # Accumulate sample weights in stable value order to locate the median
        order = np.argsort(values, kind="stable")
        ordered_values = values[order]
        ordered_weights = weights[order]
        midpoint = 0.5 * float(np.sum(ordered_weights))
        index = int(np.searchsorted(np.cumsum(ordered_weights), midpoint))
        return float(ordered_values[min(index, len(ordered_values) - 1)])

    # Measure scale-normalized sheet distances for every reference/current branch pair
    costs = np.full((branch_count, branch_count), 1e12, dtype=float)
    coverage_matrix = np.zeros((branch_count, branch_count), dtype=float)
    for reference_branch in range(branch_count):
        reference_values = reference_samples[:, reference_branch]
        for current_branch in range(branch_count):
            current_values = current_samples[:, current_branch]
            finite = (
                (reliability_weights > 0.0)
                & np.isfinite(reference_values.real)
                & np.isfinite(reference_values.imag)
                & np.isfinite(current_values.real)
                & np.isfinite(current_values.imag)
            )
            coverage = float(np.count_nonzero(finite) / max(finite.size, 1))
            coverage_matrix[reference_branch, current_branch] = coverage
            # Keep the large default cost when fewer than four usable samples overlap
            if np.count_nonzero(finite) < 4:
                continue
            reference_finite = reference_values[finite]
            current_finite = current_values[finite]
            weights = reliability_weights[finite]
            scale = max(
                weighted_median(np.abs(reference_finite), weights),
                1e-12,
            )
            normalized_distance = np.abs(
                current_finite - reference_finite
            ) / scale
            # A weighted median uses the full sheet while preventing a small
            # branch-cut neighborhood from dominating the assignment.
            costs[reference_branch, current_branch] = float(
                weighted_median(normalized_distance, weights)
                + 0.2 * (1.0 - coverage)
            )

    # Evaluate complete global permutations without changing pointwise solver continuity
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for permutation in permutations(range(branch_count)):
        total = float(
            sum(costs[index, permutation[index]] for index in range(branch_count))
        )
        candidates.append((total, tuple(int(index) for index in permutation)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    # Prefer fewer panel changes when assignment costs are numerically tied
    minimum = candidates[0][0]
    tie_tolerance = max(1e-10, 1e-8 * max(abs(minimum), 1.0))
    tied = [item for item in candidates if item[0] <= minimum + tie_tolerance]
    identity = tuple(range(branch_count))
    best_total, best_permutation = min(
        tied,
        key=lambda item: (
            sum(index != item[1][index] for index in range(branch_count)),
            item[1] != identity,
            item[1],
        ),
    )
    second_best = candidates[1][0] if len(candidates) > 1 else float("inf")
    ambiguous = len(tied) > 1

    # Apply the chosen permutation consistently to all compatible branch-indexed arrays
    kz_real = current.kz_real[:, :, best_permutation]
    kz_imag = current.kz_imag[:, :, best_permutation]
    propagating_mask = (
        current.propagating_mask[:, :, best_permutation]
        if current.propagating_mask is not None
        and current.propagating_mask.shape == current.kz_real.shape
        else current.propagating_mask
    )
    group_velocity = (
        current.vertical_group_velocity_km_s[:, :, best_permutation]
        if current.vertical_group_velocity_km_s is not None
        and current.vertical_group_velocity_km_s.shape == current.kz_real.shape
        else current.vertical_group_velocity_km_s
    )
    dispersion_residual = (
        current.normalized_dispersion_residual[:, :, best_permutation]
        if current.normalized_dispersion_residual is not None
        and current.normalized_dispersion_residual.shape == current.kz_real.shape
        else current.normalized_dispersion_residual
    )
    all_kz_roots = current.all_kz_roots
    if all_kz_roots is not None and all_kz_roots.shape == current.kz_real.shape:
        all_kz_roots = all_kz_roots[:, :, best_permutation]
    selected_root_indices = current.selected_root_indices
    if selected_root_indices is not None and selected_root_indices.shape == current.kz_real.shape:
        selected_root_indices = selected_root_indices[:, :, best_permutation]
    # Record assignment quality and preserve any existing calculation warnings
    summary = dict(current.summary)
    summary.update(
        {
            "branch_alignment_permutation": str(best_permutation),
            "branch_alignment_cost": best_total,
            "branch_alignment_second_best_cost": second_best,
            "branch_alignment_ambiguous": ambiguous,
        }
    )
    result_warnings = list(current.warnings)
    if ambiguous:
        result_warnings.append(
            "panel identity is non-unique for coincident or numerically "
            "indistinguishable k_z sheets; the prior slot order was retained"
        )
    # Build the aligned result with reference-panel order and updated branch labels
    aligned = replace(
        current,
        phase_deg=current.phase_deg[:, :, best_permutation],
        kz_real=kz_real,
        kz_imag=kz_imag,
        associated_boundaries_mhz=current.associated_boundaries_mhz[
            :, best_permutation, :
        ],
        branch_labels=_branch_labels(kz_real, current.model_key, group_velocity),
        propagating_mask=propagating_mask,
        vertical_group_velocity_km_s=group_velocity,
        normalized_dispersion_residual=dispersion_residual,
        all_kz_roots=all_kz_roots,
        selected_root_indices=selected_root_indices,
        summary=MappingProxyType(summary),
        warnings=tuple(result_warnings),
    )
    # Report the chosen permutation and the reliability/coverage of its evidence
    diagnostics = BranchAlignment(
        permutation=best_permutation,
        total_cost=best_total,
        second_best_cost=second_best,
        ambiguous=ambiguous,
        reliable_sample_fraction=float(
            np.count_nonzero(reliability_weights == 1.0)
            / max(reliability_weights.size, 1)
        ),
        valid_sample_fraction=float(
            min(
                coverage_matrix[index, best_permutation[index]]
                for index in range(branch_count)
            )
        ),
    )
    return aligned, diagnostics


def cache_info():
    '''
    Purpose
    -------
    Return hit, miss, and size statistics for the diagram-result cache.

    Inputs
    ------
    None

    Outputs
    -------
    info: functools cache-info tuple
        Cache hits, misses, configured maximum size, and current entry count.
    '''

    # Expose the cache counters without changing any entries
    return _cached_compute_diagram.cache_info()


def clear_cache() -> None:
    '''
    Purpose
    -------
    Remove all cached dispersion-diagram results.

    Inputs
    ------
    None

    Outputs
    -------
    None
        Removes all shared cached diagram results and resets cache statistics.
    '''

    # Remove retained calculations and reset LRU statistics
    _cached_compute_diagram.cache_clear()
