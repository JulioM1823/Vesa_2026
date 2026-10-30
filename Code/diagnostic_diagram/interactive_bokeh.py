'''
Purpose
-------
Bokeh server document for interactive k–omega diagnostic diagrams.

Inputs
------
None
    Uses model metadata and calculation adapters from interactive_models.

Outputs
-------
Bokeh document builders, model explorers, and plotting constants
    Each explorer keeps separate sources, callbacks, and branch-panel state.
'''

from __future__ import annotations

# Import callback helpers, numerical libraries, and Bokeh document models
from collections import OrderedDict
from html import escape
from functools import partial
from typing import Any

import cmasher as cmr
import matplotlib
import numpy as np
from bokeh.document import Document
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    Div,
    HoverTool,
    Label,
    Legend,
    LegendItem,
    LinearColorMapper,
    Range1d,
    Select,
    Slider,
    TabPanel,
    Tabs,
)
from bokeh.plotting import figure
from bokeh.themes import Theme
from matplotlib.colors import to_hex

try:  # Support package imports and the Bokeh directory-app entry point.
    from .numeric_controls import ExactSlider, SharedParameterState
    from .interactive_models import (
        DEFAULT_DOMAIN,
        DEFAULT_RESOLUTION,
        MODEL_DEFINITIONS,
        DiagramResult,
        ModelDefinition,
        align_diagram_result_branches,
        cache_info,
        compute_diagram,
        update_diagram_postprocessing,
    )
except ImportError:  # pragma: no cover - exercised by ``bokeh serve``.
    from numeric_controls import ExactSlider, SharedParameterState
    from interactive_models import (
        DEFAULT_DOMAIN,
        DEFAULT_RESOLUTION,
        MODEL_DEFINITIONS,
        DiagramResult,
        ModelDefinition,
        align_diagram_result_branches,
        cache_info,
        compute_diagram,
        update_diagram_postprocessing,
    )


# Define calculation presets and the fields available to the display selector
RESOLUTION_OPTIONS = {
    "interactive": (48, 56),
    "balanced": (64, 76),
    "detailed": (96, 120),
}
FIELD_OPTIONS = {
    "phase": "Phase difference Δφ [deg]",
    "kz_real": "Real vertical wavenumber Re(k_z) [km⁻¹]",
    "kz_imag": "Imaginary vertical wavenumber Im(k_z) [km⁻¹]",
}


def _sample_matplotlib_colormap(colormap, samples: int | None = None) -> list[str]:
    '''
    Purpose
    -------
    Translate the pipeline's Matplotlib colormap into a Bokeh palette.

    Inputs
    ------
    colormap: matplotlib.colors.Colormap
        Colormap sampled uniformly across its normalized [0, 1] interval.
    samples: int or None, optional
        Palette length; defaults to the colormap's native number of levels.

    Outputs
    -------
    palette: list of str
        Hexadecimal RGB colors; alpha channels are discarded.
    '''

    # Sample the full normalized colormap range and convert each level to RGB hex
    samples = int(colormap.N if samples is None else samples)
    return [
        to_hex(colormap(value), keep_alpha=False)
        for value in np.linspace(0.0, 1.0, samples)
    ]


# These are the exact colormaps selected by analysis.ipynb, sampled at each
# colormap's native number of levels (511 for fusion and 256 for hot).
FUSION_PALETTE = _sample_matplotlib_colormap(cmr.fusion)
HOT_PALETTE = _sample_matplotlib_colormap(matplotlib.colormaps["hot"])
MISSING_DATA_COLOR = to_hex("0.65")


class ModelExplorer:
    '''
    Purpose
    -------
    Independent controls, sources, and callback state for one model tab.

    Inputs
    ------
    doc: bokeh.document.Document
        Session document used to schedule callbacks and own the model UI.
    definition: ModelDefinition
        Model identity, exposed controls, and metadata.

    Outputs
    -------
    explorer: ModelExplorer
        Lazy per-model controls, branch plots, data sources, and callback state.
        A solve is scheduled only when the section is active.
    '''

    def __init__(self, doc: Document, definition: ModelDefinition) -> None:
        '''
        Purpose
        -------
        Create one model tab with its controls, plots, and callback state.

        Inputs
        ------
        doc: bokeh.document.Document
            Session document for callback scheduling.
        definition: ModelDefinition
            Model metadata and parameter-control definitions.

        Outputs
        -------
        None
            Initializes widget/source collections, empty branch panels, bounded
            ordered-result caching, and callbacks; leaves the model inactive/dirty.
        '''

        # Initialize independent callback, result, and panel-order state for this model
        self.doc = doc
        self.definition = definition
        self.widgets: dict[str, Slider] = {}
        self.numeric_controls = {}
        self._suppress_callbacks = False
        self._pending_callback: Any = None
        self._active = False
        self._dirty = True
        self.last_result: DiagramResult | None = None
        self._last_ordering_result: DiagramResult | None = None
        self._ordered_state_cache: OrderedDict[tuple[Any, ...], DiagramResult] = (
            OrderedDict()
        )
        self.branch_count = int(self._base_params["kz_order"])

        # Create calculation status and display controls before constructing the tab
        # layout
        self.status = Div(
            text="<span class='pending'>Open this tab to calculate the model.</span>",
            sizing_mode="stretch_width",
            styles={"padding": "8px 10px", "border-left": "4px solid #64748b"},
        )
        self.field_select = Select(
            title="Displayed field",
            value="phase",
            options=list(FIELD_OPTIONS.items()),
            width=310,
        )
        self.resolution_select = Select(
            title="Calculation grid",
            value="interactive",
            options=[
                ("interactive", "48 × 56 — interactive"),
                ("balanced", "64 × 76 — balanced"),
                ("detailed", "96 × 120 — detailed"),
            ],
            width=310,
        )
        self.show_fmode = CheckboxGroup(
            labels=["Show f-mode reference ω² = gk_h"], active=[], width=310
        )
        self.mask_warnings = CheckboxGroup(
            labels=["Mask failed or numerically unstable cells"], active=[0], width=310
        )
        self.recalculate_button = Button(
            label="Recalculate", button_type="primary", width=148
        )
        self.reset_button = Button(label="Restore defaults", width=148)

        # Build empty branch panels and controls, then connect their callbacks
        self._build_plot()
        self.controls = self._build_controls()
        self.layout = column(
            self.status,
            self.controls,
            self.plot_grid,
            sizing_mode="stretch_width",
        )
        self._wire_callbacks()

    @property
    def _base_params(self):
        '''
        Purpose
        -------
        Return the configured default parameters for this model.

        Inputs
        ------
        None
            Uses this explorer's model definition key.

        Outputs
        -------
        params: dict
            Shared config dictionary for this model; callers use it for metadata
            and do not mutate it here.
        '''

        # Support both package and notebook-style imports of the shared configuration
        try:
            from . import config
        except ImportError:  # pragma: no cover
            import config
        return config.params[self.definition.key]

    def _build_controls(self):
        '''
        Purpose
        -------
        Build grouped dispersion, diagram, and display controls for the tab.

        Inputs
        ------
        None
            Reads the parameter definitions and existing display widgets.

        Outputs
        -------
        controls: bokeh.models.Row
            Three-column layout containing exact numeric controls, diagram settings,
            implementation metadata, and display/recalculation controls. Populates
            widgets and numeric_controls on the explorer.
        '''

        # Create exact-value controls and separate solver inputs from diagram-only
        # inputs
        dispersion_widgets = []
        diagram_widgets = []
        for item in self.definition.parameters:
            widget = Slider(
                title=item.title,
                start=item.minimum,
                end=item.maximum,
                step=item.step,
                value=item.default,
                format="0.00000000" if item.key in {"N2", "wac"} else "0[.]000000",
                width=310,
                name=f"{self.definition.key}:{item.key}",
            )
            self.widgets[item.key] = widget
            target = dispersion_widgets if item.role == "dispersion" else diagram_widgets
            control = ExactSlider(widget, item.key)
            widget.width = 205
            self.numeric_controls[item.key] = control
            target.append(control.layout)

        # Display model implementation metadata when it is present
        metadata = []
        if self.definition.metadata_note:
            metadata.append(
                Div(
                    text=(
                        "<div style='background:#fff7ed;border:1px solid #fed7aa;"
                        "padding:8px;border-radius:4px'><b>Implementation note.</b> "
                        f"{escape(self.definition.metadata_note)}</div>"
                    ),
                    width=310,
                )
            )

        # Arrange model information, diagram settings, and display controls in three
        # columns
        identity_column = column(
            Div(
                text=(
                    f"<h2 style='margin:0'>{escape(self.definition.title)}</h2>"
                    f"<p style='margin:4px 0 10px;color:#475569'>{escape(self.definition.reference)}</p>"
                ),
                width=310,
            ),
            Div(text="<h3 style='margin:6px 0'>Dispersion-relation parameters</h3>"),
            *dispersion_widgets,
            width=330,
        )
        diagram_column = column(
            Div(text="<h3 style='margin:12px 0 6px'>Diagram and reference</h3>"),
            *diagram_widgets,
            *metadata,
            width=330,
        )
        display_column = column(
            Div(text="<h3 style='margin:12px 0 6px'>Display</h3>"),
            self.field_select,
            self.resolution_select,
            self.show_fmode,
            self.mask_warnings,
            row(self.recalculate_button, self.reset_button),
            Div(
                text=(
                    "<p style='margin:6px 0 0;color:#475569'>Scroll normally over "
                    "the plots to move between solution rows. Select <b>Wheel Zoom</b> "
                    "in a plot toolbar when you want the wheel to zoom that plot.</p>"
                ),
                width=310,
            ),
            width=330,
        )
        return row(
            identity_column,
            diagram_column,
            display_column,
            sizing_mode="stretch_width",
            styles={"padding": "4px 12px 12px 4px"},
        )

    def _build_plot(self) -> None:
        '''
        Purpose
        -------
        Create branch plots, image sources, hover tools, and references.

        Inputs
        ------
        None
            Uses configured branch/boundary counts and the default display domain.

        Outputs
        -------
        None
            Creates shared axes/color mapping, one image and hover renderer per
            branch, boundary/reference sources, legends, color bars, and message
            labels. Stores the grid plus first-panel compatibility aliases.
        '''

        # Share physical axes and one color mapper across all solution panels
        self.shared_x_range = Range1d(DEFAULT_DOMAIN[0], DEFAULT_DOMAIN[1])
        self.shared_y_range = Range1d(DEFAULT_DOMAIN[2], DEFAULT_DOMAIN[3])
        self.mapper = LinearColorMapper(
            palette=FUSION_PALETTE,
            low=-80.0,
            high=80.0,
            nan_color=MISSING_DATA_COLOR,
        )
        # Create empty hover columns for every field and every solution branch
        hover_data: dict[str, Any] = {
            "kh": [], "nu": [], "warning": [], "mode_ambiguous": []
        }
        for branch in range(self.branch_count):
            hover_data.update(
                {
                    f"phase_{branch}": [],
                    f"kzr_{branch}": [],
                    f"kzi_{branch}": [],
                    f"field_{branch}": [],
                    f"valid_{branch}": [],
                    f"vgz_{branch}": [],
                    f"propagation_{branch}": [],
                    f"residual_{branch}": [],
                }
            )
        self.hover_source = ColumnDataSource(
            data=hover_data, name=f"{self.definition.key}:hover_source"
        )
        self.fmode_source = ColumnDataSource(data={"kh": [], "nu": []})

        # Prepare collections that retain all branch-specific Bokeh models
        self.plots = []
        self.image_sources = []
        self.image_renderers = []
        self.hover_renderers = []
        self.fmode_renderers = []
        self.boundary_sources_by_branch = []
        self.boundary_renderers_by_branch = []
        self.legends = []
        self.color_bars = []
        self.message_labels = []
        omega_order = int(self._base_params["omega_order"])

        # Construct a complete image, hover overlay, references, and annotations per
        # branch
        for branch in range(self.branch_count):
            image_source = ColumnDataSource(
                data={
                    "image": [np.full((2, 2), np.nan)],
                    "x": [DEFAULT_DOMAIN[0]],
                    "y": [DEFAULT_DOMAIN[2]],
                    "dw": [DEFAULT_DOMAIN[1] - DEFAULT_DOMAIN[0]],
                    "dh": [DEFAULT_DOMAIN[3] - DEFAULT_DOMAIN[2]],
                },
                name=f"{self.definition.key}:image_source:{branch}",
            )
            plot = figure(
                title=f"Solution {branch + 1}",
                width=700,
                height=700,
                x_range=self.shared_x_range,
                y_range=self.shared_y_range,
                x_axis_label="kₕ [Mm⁻¹]",
                y_axis_label="ν [mHz]",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                # Leave wheel zoom available without capturing ordinary page scrolling.
                # Users can explicitly activate it from the toolbar when desired.
                active_scroll=None,
                toolbar_location="above",
                name=f"{self.definition.key}:plot:{branch}",
            )
            # Match the notebook plot typography and axis styling
            plot.grid.visible = False
            plot.background_fill_color = "white"
            plot.border_fill_color = "white"
            plot.outline_line_color = "black"
            plot.title.text_font = "Times New Roman"
            plot.title.text_font_size = "16pt"
            plot.title.text_font_style = "normal"
            for axis in (plot.xaxis[0], plot.yaxis[0]):
                axis.axis_label_text_font = "Times New Roman"
                axis.axis_label_text_font_size = "16pt"
                axis.axis_label_text_font_style = "normal"
                axis.major_label_text_font = "Times New Roman"
                axis.major_label_text_font_size = "16pt"
                axis.major_tick_in = 6
                axis.major_tick_out = 0
                axis.minor_tick_in = 3
                axis.minor_tick_out = 0

            # Render the selected field and add invisible cells for numerical hover
            # details
            image_renderer = plot.image(
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                source=image_source,
                color_mapper=self.mapper,
                level="image",
            )
            hover_renderer = plot.rect(
                x="kh",
                y="nu",
                width=(DEFAULT_DOMAIN[1] - DEFAULT_DOMAIN[0]) / DEFAULT_RESOLUTION[0],
                height=(DEFAULT_DOMAIN[3] - DEFAULT_DOMAIN[2]) / DEFAULT_RESOLUTION[1],
                source=self.hover_source,
                fill_alpha=0.0,
                line_alpha=0.0,
            )
            plot.add_tools(
                HoverTool(
                    renderers=[hover_renderer],
                    tooltips=[
                        ("solution", f"Solution {branch + 1}"),
                        ("k_h", "@kh{0.000} Mm⁻¹"),
                        ("ν", "@nu{0.000} mHz"),
                        ("Δφ", f"@phase_{branch}{{0.000}} deg"),
                        ("Re(k_z)", f"@kzr_{branch}{{0.000000}} km⁻¹"),
                        ("Im(k_z)", f"@kzi_{branch}{{0.000000}} km⁻¹"),
                        ("displayed value", f"@field_{branch}{{0.000000}}"),
                        ("finite", f"@valid_{branch}"),
                        ("numerical warning", "@warning"),
                        ("vertical group velocity", f"@vgz_{branch}{{0.000}} km s⁻¹"),
                        ("energy propagation", f"@propagation_{branch}"),
                        ("normalized dispersion residual", f"@residual_{branch}{{0.0e}}"),
                        ("mode ambiguous", "@mode_ambiguous"),
                    ],
                    mode="mouse",
                )
            )
            # Create the optional f-mode reference and initially hidden dispersion
            # boundaries
            fmode_renderer = plot.line(
                "kh",
                "nu",
                source=self.fmode_source,
                line_color="gray",
                line_dash="solid",
                line_width=1.5,
            )
            boundary_sources = []
            boundary_renderers = []
            for boundary in range(omega_order):
                source = ColumnDataSource(data={"kh": [], "nu": []})
                renderer = plot.line(
                    "kh",
                    "nu",
                    source=source,
                    line_color="black",
                    line_dash="solid",
                    line_width=1.5,
                    visible=False,
                )
                boundary_sources.append(source)
                boundary_renderers.append(renderer)

            # Add a per-panel legend and color bar using the shared color mapper
            legend = Legend(
                items=[],
                location="top_left",
                click_policy="hide",
                background_fill_alpha=0.82,
                label_text_font="Times New Roman",
                label_text_font_size="14pt",
            )
            plot.add_layout(legend)
            color_bar = ColorBar(
                color_mapper=self.mapper,
                ticker=BasicTicker(),
                title="Δφ [deg]",
                label_standoff=7,
                title_text_font="Times New Roman",
                title_text_font_size="16pt",
                title_text_font_style="normal",
                major_label_text_font="Times New Roman",
                major_label_text_font_size="16pt",
            )
            plot.add_layout(color_bar, "right")
            # Reserve a centered message for panels without finite displayed values
            message_label = Label(
                x=4.0,
                y=5.5,
                text="",
                text_align="center",
                text_baseline="middle",
                text_font_size="14px",
                text_color="#991b1b",
                background_fill_color="white",
                background_fill_alpha=0.92,
                border_line_color="#991b1b",
                border_line_alpha=0.7,
                visible=False,
            )
            plot.add_layout(message_label)

            # Retain every model reference for later source and presentation updates
            self.plots.append(plot)
            self.image_sources.append(image_source)
            self.image_renderers.append(image_renderer)
            self.hover_renderers.append(hover_renderer)
            self.fmode_renderers.append(fmode_renderer)
            self.boundary_sources_by_branch.append(boundary_sources)
            self.boundary_renderers_by_branch.append(boundary_renderers)
            self.legends.append(legend)
            self.color_bars.append(color_bar)
            self.message_labels.append(message_label)

        # Arrange six-branch models in three columns and other models in two
        columns = 3 if self.branch_count == 6 else 2
        plot_rows = [
            self.plots[index : index + columns]
            for index in range(0, self.branch_count, columns)
        ]
        self.plot_grid = gridplot(
            plot_rows,
            merge_tools=False,
            toolbar_location=None,
        )

        # Singular aliases retain a small compatibility surface for notebooks
        # that inspected the first panel directly.
        self.plot = self.plots[0]
        self.image_source = self.image_sources[0]
        self.image_renderer = self.image_renderers[0]
        self.hover_renderer = self.hover_renderers[0]
        self.fmode_renderer = self.fmode_renderers[0]
        self.boundary_sources = self.boundary_sources_by_branch[0]
        self.boundary_renderers = self.boundary_renderers_by_branch[0]
        self.legend = self.legends[0]
        self.color_bar = self.color_bars[0]
        self.message_label = self.message_labels[0]

    def _wire_callbacks(self) -> None:
        '''
        Purpose
        -------
        Connect control changes and buttons to this explorer's handlers.

        Inputs
        ------
        None
            Uses widgets and controls already created on this explorer.

        Outputs
        -------
        None
            Registers model-parameter, display, resolution, and button handlers.
        '''

        # Attach parameter handlers with each control identity and recomputation
        # category
        for item in self.definition.parameters:
            widget = self.widgets[item.key]
            widget.on_change(
                "value",
                partial(self._parameter_changed, item.key, item.role),
            )
        # Connect calculation, presentation, and button controls to their own handlers
        self.resolution_select.on_change("value", self._calculation_changed)
        self.field_select.on_change("value", self._display_changed)
        self.show_fmode.on_change("active", self._display_changed)
        self.mask_warnings.on_change("active", self._display_changed)
        self.recalculate_button.on_click(self._recalculate_clicked)
        self.reset_button.on_click(self._restore_defaults)

    def set_active(self, active: bool) -> None:
        '''
        Purpose
        -------
        Activate lazy calculation for this tab and cancel pending inactive work.

        Inputs
        ------
        active: bool
            Whether this model is the currently displayed tab.

        Outputs
        -------
        None
            Cancels queued timeout work when inactive. Activating a dirty model
            schedules a recalculation after 20 ms.
        '''

        # Cancel pending work on deactivation and calculate dirty state on activation
        self._active = bool(active)
        if not self._active and self._pending_callback is not None:
            try:
                self.doc.remove_timeout_callback(self._pending_callback)
            except ValueError:
                pass
            self._pending_callback = None
        if self._active and self._dirty:
            self.schedule_recompute(delay_ms=20)

    def _parameter_changed(
        self, key: str, role: str, attr: str, old: Any, new: Any
    ) -> None:
        '''
        Purpose
        -------
        Update diagram-only state or schedule a fresh dispersion calculation.

        Inputs
        ------
        key, role: str
            Changed parameter identifier and its dispersion/diagram category.
        attr: str
            Bokeh property name passed by the callback.
        old, new: Any
            Previous and updated widget values supplied by Bokeh.

        Outputs
        -------
        None
            Applies diagram-only updates to a clean active result; otherwise marks
            the model dirty and schedules a debounced solve only for active tabs.
            Invalid postprocessing clears the plots and reports the error.
        '''

        # Ignore programmatic widget changes while a grouped update is in progress
        if self._suppress_callbacks:
            return
        # Reuse roots for diagram-only edits when the active result is already current
        if role == "diagram" and not self._dirty and self.last_result is not None and self._active:
            try:
                self.last_result = update_diagram_postprocessing(
                    self.last_result, self._current_values()
                )
                self._dirty = False
                self._update_display(self.last_result)
                self._last_ordering_result = self.last_result
                self._remember_ordered_result(self.last_result)
                self._set_status(
                    "Diagram-only quantity updated without repeating the root solve.",
                    "ok",
                )
                return
            except Exception as exc:
                self._clear_plot(f"No valid solution: {exc}")
                self._set_status(f"<b>Calculation failed:</b> {escape(str(exc))}", "error")
                self._dirty = True
                return
        # Defer inactive calculations and debounce active solver-input changes
        self._dirty = True
        if self._active:
            self.schedule_recompute(delay_ms=250)
        else:
            self._set_status(
                "Parameters changed; the model will recalculate when this tab is opened.",
                "pending",
            )

    def _calculation_changed(self, attr: str, old: Any, new: Any) -> None:
        '''
        Purpose
        -------
        Mark the model dirty when its calculation resolution changes.

        Inputs
        ------
        attr: str
            Changed Bokeh property name.
        old, new: Any
            Previous and selected resolution-option values.

        Outputs
        -------
        None
            Routes the resolution change through the dispersion-change handler.
        '''

        # Treat a resolution change as a new dispersion-grid calculation
        self._parameter_changed("resolution", "dispersion", attr, old, new)

    def _display_changed(self, attr: str, old: Any, new: Any) -> None:
        '''
        Purpose
        -------
        Refresh plot presentation without recalculating the dispersion relation.

        Inputs
        ------
        attr: str
            Changed Bokeh property name.
        old, new: Any
            Previous and updated field/checkbox values.

        Outputs
        -------
        None
            Refreshes existing result sources unless callbacks are suppressed;
            does not solve the dispersion relation.
        '''

        # Refresh presentation only when a reusable result is available
        if not self._suppress_callbacks and self.last_result is not None:
            self._update_display(self.last_result)

    def _recalculate_clicked(self) -> None:
        '''
        Purpose
        -------
        Request an immediate recalculation for the active tab.

        Inputs
        ------
        None

        Outputs
        -------
        None
            Marks the model dirty and requests an immediate active-tab solve.
        '''

        # Mark the current result stale and request one immediate active-tab calculation
        self._dirty = True
        self.schedule_recompute(delay_ms=0)

    def _restore_defaults(self) -> None:
        '''
        Purpose
        -------
        Reset model and display controls, then recalculate the default result.

        Inputs
        ------
        None

        Outputs
        -------
        None
            Resets shared state when available, restores parameter/display defaults
            while suppressing callbacks, and schedules one active-tab recalculation.
        '''

        # Restore shared physical defaults before resetting this tab's display controls
        if hasattr(self, "parameter_state"):
            self.parameter_state.reset(self)
        # Suppress individual widget callbacks so resetting creates a single calculation
        # request
        self._suppress_callbacks = True
        try:
            for item in self.definition.parameters:
                self.widgets[item.key].value = item.default
            self.field_select.value = "phase"
            self.resolution_select.value = "interactive"
            self.show_fmode.active = []
            self.mask_warnings.active = [0]
        finally:
            self._suppress_callbacks = False
        self._dirty = True
        self.schedule_recompute(delay_ms=0)

    def schedule_recompute(self, delay_ms: int = 250) -> None:
        '''
        Purpose
        -------
        Debounce a recalculation request on the active Bokeh document.

        Inputs
        ------
        delay_ms: int, optional
            Timeout delay in milliseconds; negative values are clamped to zero.

        Outputs
        -------
        None
            Replaces any queued timeout with one recompute callback. Inactive tabs
            return without scheduling work.
        '''

        # Keep inactive tabs lazy and replace any prior debounce timeout
        if not self._active:
            return
        if self._pending_callback is not None:
            try:
                self.doc.remove_timeout_callback(self._pending_callback)
            except ValueError:
                pass
            self._pending_callback = None
        # Schedule the recompute after the requested nonnegative delay
        self._pending_callback = self.doc.add_timeout_callback(
            self._recompute, max(0, int(delay_ms))
        )

    def _current_values(self) -> dict[str, float]:
        '''
        Purpose
        -------
        Read the current numeric values from all model parameter widgets.

        Inputs
        ------
        None
            Reads the current model-parameter widgets.

        Outputs
        -------
        values: dict[str, float]
            Parameter values in displayed units. A missing widget value raises
            ValueError before a calculation is attempted.
        '''

        # Read all control values in display units and reject missing numeric inputs
        values: dict[str, float] = {}
        for key, widget in self.widgets.items():
            if widget.value is None:
                raise ValueError(f"{widget.title} requires a numeric value.")
            values[key] = float(widget.value)
        return values

    @staticmethod
    def _result_state_key(result: DiagramResult) -> tuple[Any, ...]:
        '''
        Purpose
        -------
        Build a stable cache key from model, parameter values, and grid dimensions.

        Inputs
        ------
        result: DiagramResult
            Model result whose normalized state is used for panel-order caching.

        Outputs
        -------
        key: tuple
            Model identifier, parameter-value tuple, kh-axis length, and frequency-
            axis length; axis coordinates are not included in this local UI key.
        '''

        # Identify local panel-order states by model, normalized inputs, and grid
        # dimensions
        return (
            result.model_key,
            result.parameter_values,
            len(result.kh_mm_inv),
            len(result.nu_mhz),
        )

    def _remember_ordered_result(self, result: DiagramResult) -> None:
        '''
        Purpose
        -------
        Store a panel-ordered result in the bounded recency cache.

        Inputs
        ------
        result: DiagramResult
            Result whose branch axis already follows the desired panel identities.

        Outputs
        -------
        None
            Moves the state to the most-recent position and evicts oldest entries
            until at most 24 ordered results remain.
        '''

        # Refresh recency before evicting the least recently used ordered states
        key = self._result_state_key(result)
        self._ordered_state_cache.pop(key, None)
        self._ordered_state_cache[key] = result
        while len(self._ordered_state_cache) > 24:
            self._ordered_state_cache.popitem(last=False)

    def _order_result_for_panels(self, result: DiagramResult):
        '''
        Purpose
        -------
        Restore cached branch identities or align a new result to the prior solve.

        Inputs
        ------
        result: DiagramResult
            Newly obtained solver result before restoring or aligning panel order.

        Outputs
        -------
        ordered: DiagramResult
            Cached panel-ordered result or globally aligned new result.
        alignment: BranchAlignment or None
            Assignment diagnostics when matching to a previous result was needed.
        cache_hit: bool
            Whether an already ordered result was restored from the local cache.
        '''

        # Restore an exact cached panel ordering before attempting a new global match
        key = self._result_state_key(result)
        cached = self._ordered_state_cache.pop(key, None)
        if cached is not None:
            self._ordered_state_cache[key] = cached
            return cached, None, True
        # Align new branch sheets with the previous result and remember the chosen order
        alignment = None
        if self._last_ordering_result is not None:
            result, alignment = align_diagram_result_branches(
                result, self._last_ordering_result
            )
        self._remember_ordered_result(result)
        return result, alignment, False

    def _recompute(self) -> None:
        '''
        Purpose
        -------
        Show a busy state and defer synchronous solving to the next tick.

        Inputs
        ------
        None
            Invoked by the document's pending timeout callback.

        Outputs
        -------
        None
            Clears the timeout handle, marks inactive work dirty, or displays busy
            status and schedules the synchronous solve for the next document tick.
        '''

        # Clear the completed timeout handle and recheck tab activity before solving
        self._pending_callback = None
        if not self._active:
            self._dirty = True
            return
        self._set_status("Calculating exact polynomial roots and tracking branches…", "busy")
        # Return once so the server can paint the busy state before a detailed
        # synchronous solve occupies this session's document lock.
        self.doc.add_next_tick_callback(self._perform_recompute)

    def _perform_recompute(self) -> None:
        '''
        Purpose
        -------
        Solve the current model, align branch panels, and report result quality.

        Inputs
        ------
        None
            Reads current parameter controls and the selected resolution.

        Outputs
        -------
        None
            Computes/caches roots, restores global panel identities, updates sources,
            and reports timing and numerical quality. Calculation exceptions clear
            this explorer's plots and leave its state dirty.
        '''

        # Skip a next-tick solve if the user switched to another tab in the meantime
        if not self._active:
            self._dirty = True
            return
        # Measure cache reuse while solving the current physical state and resolution
        before = cache_info()
        try:
            raw_result = compute_diagram(
                self.definition.key,
                self._current_values(),
                RESOLUTION_OPTIONS[self.resolution_select.value],
            )
            after = cache_info()
            cache_hit = after.hits > before.hits
            # Restore branch-panel identities before replacing visible fields
            result, alignment, ordering_cache_hit = self._order_result_for_panels(
                raw_result
            )
            self.last_result = result
            self._last_ordering_result = result
            self._dirty = False
            self._update_display(result)
            # Format calculation timing, root residuals, and panel-assignment
            # information
            timing = "cache hit" if cache_hit else f"{result.elapsed_seconds:.2f} s"
            residual = result.summary["max_kz_root_residual"]
            residual_text = (
                f"; max normalized k_z residual {residual:.2e}"
                if np.isfinite(residual)
                else ""
            )
            alignment_text = ""
            if ordering_cache_hit:
                alignment_text = "; restored cached panel identities"
            elif alignment is not None:
                alignment_text = (
                    "; globally matched all solution sheets"
                    f" (cost {alignment.total_cost:.3g})"
                )
            # Report numerical diagnostics without treating warnings as calculation
            # failures
            if result.warnings:
                details = "; ".join(escape(text) for text in result.warnings)
                self._set_status(
                    f"Updated in {timing}{residual_text}{alignment_text}. "
                    f"<b>Warnings:</b> {details}.",
                    "warning",
                )
            else:
                self._set_status(
                    f"Updated in {timing}{residual_text}{alignment_text}; all "
                    "sampled roots are finite and stable.",
                    "ok",
                )
        except Exception as exc:  # A bad model choice must not break other tabs.
            # Isolate calculation failures to this tab and remove its stale plotted data
            self.last_result = None
            self._dirty = True
            self._clear_plot(f"No valid solution: {exc}")
            self._set_status(f"<b>Calculation failed:</b> {escape(str(exc))}", "error")

    @staticmethod
    def _pipeline_kz_limits(values: np.ndarray) -> tuple[float, float]:
        '''
        Purpose
        -------
        Match ``kz_color_limits(..., robust=False)`` from the pipeline.

        Inputs
        ------
        values: np.ndarray
            Root component values in km^-1; nonfinite entries are excluded.

        Outputs
        -------
        low, high: tuple of float
            Finite global extrema, padded for a constant array. Returns (0, 1)
            when no finite values exist.
        '''

        # Exclude unavailable roots and provide a fallback interval for empty fields
        finite = values[np.isfinite(values)]
        if not finite.size:
            return 0.0, 1.0
        low, high = float(np.min(finite)), float(np.max(finite))
        # Pad a constant finite field so the linear color scale has nonzero extent
        if np.isclose(low, high):
            padding = max(abs(low) * 1e-9, 1e-12)
            return low - padding, high + padding
        return low, high

    def _field(self, result: DiagramResult, branch: int) -> np.ndarray:
        '''
        Purpose
        -------
        Select the currently displayed phase, real-kz, or imaginary-kz field.

        Inputs
        ------
        result: DiagramResult
            Result containing phase and root-component arrays.
        branch: int
            Zero-based returned branch index.

        Outputs
        -------
        values: np.ndarray
            Selected two-dimensional field (n_omega, n_kh): degrees for phase,
            or km^-1 for a real/imaginary vertical-wavenumber component.
        '''

        # Select one branch of the requested phase or vertical-wavenumber component
        if self.field_select.value == "phase":
            return result.phase_deg[:, :, branch]
        if self.field_select.value == "kz_real":
            return result.kz_real[:, :, branch]
        return result.kz_imag[:, :, branch]

    def _update_display(self, result: DiagramResult) -> None:
        '''
        Purpose
        -------
        Update branch images, hover data, color mapping, and plot annotations.

        Inputs
        ------
        result: DiagramResult
            Result with exactly this explorer's configured number of branches.

        Outputs
        -------
        None
            Replaces image/hover/reference data, applies display masking and color
            limits, and refreshes labels, boundaries, legends, and empty-data
            messages. A branch-count mismatch raises ValueError.
        '''

        # Validate panel count and prepare shared physical coordinates and warning flags
        if result.phase_deg.shape[2] != self.branch_count:
            raise ValueError("The solver returned an unexpected number of solutions.")
        kh_grid, nu_grid = np.meshgrid(result.kh_mm_inv, result.nu_mhz)
        warning = result.numerical_warning_mask.copy()
        is_mhd_model = result.model_key in {"alfven_mode", "slow_mode", "fast_mode"}
        if is_mhd_model and result.mode_ambiguous_mask is not None:
            warning = warning | result.mode_ambiguous_mask
        # Flatten common cell coordinates and masks for the hover data source
        hover_data: dict[str, Any] = {
            "kh": kh_grid.ravel(),
            "nu": nu_grid.ravel(),
            "warning": np.where(warning, "yes", "no").ravel(),
            "mode_ambiguous": np.where(
                result.mode_ambiguous_mask
                if is_mhd_model and result.mode_ambiguous_mask is not None
                else np.zeros(warning.shape, dtype=bool),
                "yes",
                "no",
            ).ravel(),
        }

        # Use the notebook phase range or finite root extrema for the selected field
        if self.field_select.value == "phase":
            self.mapper.palette = FUSION_PALETTE
            self.mapper.low, self.mapper.high = -80.0, 80.0
            colorbar_title = "Δφ [deg]"
        elif self.field_select.value == "kz_real":
            self.mapper.palette = HOT_PALETTE
            self.mapper.low, self.mapper.high = self._pipeline_kz_limits(
                result.kz_real
            )
            colorbar_title = "k_z [km⁻¹]"
        else:
            self.mapper.palette = HOT_PALETTE
            self.mapper.low, self.mapper.high = self._pipeline_kz_limits(
                result.kz_imag
            )
            colorbar_title = "k_z [km⁻¹]"
        for color_bar in self.color_bars:
            color_bar.title = colorbar_title

        # Refresh the f-mode curve and physical cell dimensions used by hover overlays
        self.fmode_source.data = {"kh": result.kh_mm_inv, "nu": result.fmode_nu_mhz}
        cell_width = (DEFAULT_DOMAIN[1] - DEFAULT_DOMAIN[0]) / max(
            len(result.kh_mm_inv) - 1, 1
        )
        cell_height = (DEFAULT_DOMAIN[3] - DEFAULT_DOMAIN[2]) / max(
            len(result.nu_mhz) - 1, 1
        )
        # Replace each panel image and apply numerical masking only to its displayed
        # copy
        for branch in range(self.branch_count):
            displayed = self._field(result, branch).copy()
            if 0 in self.mask_warnings.active:
                displayed[warning] = np.nan
            self.image_sources[branch].data = {
                "image": [displayed],
                "x": [DEFAULT_DOMAIN[0]],
                "y": [DEFAULT_DOMAIN[2]],
                "dw": [DEFAULT_DOMAIN[1] - DEFAULT_DOMAIN[0]],
                "dh": [DEFAULT_DOMAIN[3] - DEFAULT_DOMAIN[2]],
            }
            # Retain unmasked phase/root values and optional propagation diagnostics for
            # hover
            phase = result.phase_deg[:, :, branch]
            kz_real = result.kz_real[:, :, branch]
            kz_imag = result.kz_imag[:, :, branch]
            finite = np.isfinite(kz_real) & np.isfinite(kz_imag)
            vgz = (
                result.vertical_group_velocity_km_s[:, :, branch]
                if result.vertical_group_velocity_km_s is not None
                and result.vertical_group_velocity_km_s.shape == result.kz_real.shape
                else np.full(kz_real.shape, np.nan)
            )
            propagating = (
                result.propagating_mask[:, :, branch]
                if result.propagating_mask is not None
                and result.propagating_mask.shape == result.kz_real.shape
                else np.zeros(kz_real.shape, dtype=bool)
            )
            residual = (
                result.normalized_dispersion_residual[:, :, branch]
                if result.normalized_dispersion_residual is not None
                and result.normalized_dispersion_residual.shape == result.kz_real.shape
                else np.full(kz_real.shape, np.nan)
            )
            # Add this branch's values and propagation classifications to the shared
            # hover table
            hover_data.update(
                {
                    f"phase_{branch}": phase.ravel(),
                    f"kzr_{branch}": kz_real.ravel(),
                    f"kzi_{branch}": kz_imag.ravel(),
                    f"field_{branch}": displayed.ravel(),
                    f"valid_{branch}": np.where(finite, "yes", "no").ravel(),
                    f"vgz_{branch}": vgz.ravel(),
                    f"propagation_{branch}": np.where(
                        propagating, "propagating", np.where(finite, "evanescent", "undefined")
                    ).ravel(),
                    f"residual_{branch}": residual.ravel(),
                }
            )
            # Update cell geometry, panel names, reference visibility, and associated
            # boundaries
            self.hover_renderers[branch].glyph.width = cell_width
            self.hover_renderers[branch].glyph.height = cell_height
            label = result.branch_labels[branch].replace(
                f"Tracked k_z branch {branch + 1}", f"Solution {branch + 1}"
            )
            self.plots[branch].title.text = label
            self.fmode_renderers[branch].visible = 0 in self.show_fmode.active
            self._update_boundaries(result, branch)

            # Rebuild legends from currently visible reference and boundary renderers
            legend_items = []
            if self.fmode_renderers[branch].visible:
                legend_items.append(
                    LegendItem(
                        label="f-mode reference",
                        renderers=[self.fmode_renderers[branch]],
                    )
                )
            for index, renderer in enumerate(
                self.boundary_renderers_by_branch[branch]
            ):
                if not renderer.visible:
                    continue
                if self.definition.key == "slow_mode":
                    boundary_label = "Slow magnetoacoustic boundary"
                elif self.definition.key == "fast_mode":
                    boundary_label = "Fast magnetoacoustic boundary"
                else:
                    boundary_label = f"Associated dispersion boundary {index + 1}"
                legend_items.append(
                    LegendItem(label=boundary_label, renderers=[renderer])
                )
            self.legends[branch].items = legend_items
            # Show an explanatory label when the selected field contains no finite
            # values
            self.message_labels[branch].visible = not np.any(np.isfinite(displayed))
            self.message_labels[branch].text = (
                "No finite values for this field and solution"
                if self.message_labels[branch].visible
                else ""
            )
        # Publish the complete equal-length hover columns after all branches are
        # prepared
        self.hover_source.data = hover_data

    def _update_boundaries(self, result: DiagramResult, branch: int) -> None:
        '''
        Purpose
        -------
        Refresh the dispersion-boundary sources associated with one branch.

        Inputs
        ------
        result: DiagramResult
            Calculation with boundary frequencies in mHz.
        branch: int
            Zero-based panel whose boundary sources are refreshed.

        Outputs
        -------
        None
            Updates kh (Mm^-1) and frequency (mHz) line sources, hiding all-NaN
            curves. The legacy combined model uses semantic slow/fast boundaries;
            other models use curves associated with the selected kz branch.
        '''

        # Resolve the persistent sources and renderers belonging to this panel
        boundary_sources = self.boundary_sources_by_branch[branch]
        boundary_renderers = self.boundary_renderers_by_branch[branch]
        # Map legacy combined-model curves by slow/fast semantic labels
        if self.definition.key == "slow_fast_mode":
            label_to_index = {
                label: index for index, label in enumerate(result.boundary_labels)
            }
            semantic_labels = (
                "Slow magnetoacoustic branch",
                "Fast magnetoacoustic branch",
            )
            indices = [label_to_index.get(label, -1) for label in semantic_labels]
            for slot in range(len(boundary_sources)):
                index = indices[slot] if slot < len(indices) else -1
                values = (
                    result.omega_boundaries_mhz[:, index]
                    if index >= 0
                    else np.full_like(result.kh_mm_inv, np.nan)
                )
                boundary_sources[slot].data = {
                    "kh": result.kh_mm_inv,
                    "nu": values,
                }
                boundary_renderers[slot].visible = bool(np.any(np.isfinite(values)))
            return

        # Plot only the boundary curves associated with this tracked vertical-root
        # branch
        for index, source in enumerate(boundary_sources):
            values = result.associated_boundaries_mhz[:, branch, index]
            source.data = {"kh": result.kh_mm_inv, "nu": values}
            boundary_renderers[index].visible = bool(np.any(np.isfinite(values)))

    def _clear_plot(self, message: str) -> None:
        '''
        Purpose
        -------
        Clear plot sources and show a message in every branch panel.

        Inputs
        ------
        message: str
            Text displayed in every branch panel after clearing its sources.

        Outputs
        -------
        None
            Replaces images with NaN placeholders, empties hover/reference/boundary
            data, hides boundary lines, and clears legends.
        '''

        # Replace displayed arrays with a NaN placeholder while retaining plot models
        for image_source in self.image_sources:
            image_source.data = {
                "image": [np.full((2, 2), np.nan)],
                "x": [DEFAULT_DOMAIN[0]],
                "y": [DEFAULT_DOMAIN[2]],
                "dw": [DEFAULT_DOMAIN[1] - DEFAULT_DOMAIN[0]],
                "dh": [DEFAULT_DOMAIN[3] - DEFAULT_DOMAIN[2]],
            }
        # Clear shared hover/reference data and hide stale boundary curves and legends
        self.hover_source.data = {key: [] for key in self.hover_source.data}
        self.fmode_source.data = {"kh": [], "nu": []}
        for sources, renderers, legend, label in zip(
            self.boundary_sources_by_branch,
            self.boundary_renderers_by_branch,
            self.legends,
            self.message_labels,
        ):
            for source, renderer in zip(sources, renderers):
                source.data = {"kh": [], "nu": []}
                renderer.visible = False
            legend.items = []
            label.text = message
            label.visible = True

    def _set_status(self, message: str, state: str) -> None:
        '''
        Purpose
        -------
        Set the status text and accent color for the supplied state.

        Inputs
        ------
        message: str
            Status HTML; callers escape dynamic text before passing it here.
        state: str
            One of 'pending', 'busy', 'ok', 'warning', or 'error'.

        Outputs
        -------
        None
            Updates status text and accent styling; an unknown state raises KeyError.
        '''

        # Select the state accent and publish the supplied status HTML
        colors = {
            "pending": "#64748b",
            "busy": "#2563eb",
            "ok": "#15803d",
            "warning": "#b45309",
            "error": "#b91c1c",
        }
        color = colors[state]
        self.status.styles = {
            "padding": "8px 10px",
            "border-left": f"4px solid {color}",
            "background": "#f8fafc",
        }
        self.status.text = message


def _theme() -> Theme:
    '''
    Purpose
    -------
    Create the shared light Bokeh theme used by the document.

    Inputs
    ------
    None

    Outputs
    -------
    theme: bokeh.themes.Theme
        Fresh light theme setting figure background and outline colors.
    '''

    # Construct a fresh light figure theme for this Bokeh session
    return Theme(
        json={
            "attrs": {
                "figure": {"background_fill_color": "white", "outline_line_color": "black"},
            }
        }
    )


def build_document(doc: Document, *, schedule_initial: bool = True) -> list[ModelExplorer]:
    '''
    Purpose
    -------
    Populate a Bokeh document and return its independently scoped sections.

    Inputs
    ------
    doc: bokeh.document.Document
        Session document receiving the application root and theme.
    schedule_initial: bool, optional
        Schedule the first active model's calculation after 20 ms when True.

    Outputs
    -------
    sections: list of ModelExplorer
        One explorer per declared model, connected by shared physical parameter
        state. Mutates doc title/theme/roots, wires tab activation, and marks
        the first model active.
    '''

    # Set session metadata and create one independent explorer for every visible model
    doc.title = "Interactive k–ω Dispersion Explorer"
    doc.theme = _theme()
    sections = [ModelExplorer(doc, definition) for definition in MODEL_DEFINITIONS]
    # Connect common physical parameters and wrap model layouts in tab panels
    SharedParameterState(sections)
    panels = [
        TabPanel(child=section.layout, title=section.definition.title)
        for section in sections
    ]
    tabs = Tabs(tabs=panels, active=0, sizing_mode="stretch_width", name="model_tabs")

    def active_changed(attr: str, old: int, new: int) -> None:
        '''
        Purpose
        -------
        Activate only the selected model tab and leave other tabs lazy.

        Inputs
        ------
        attr: str
            Changed Tabs property name.
        old, new: int
            Previous and selected zero-based tab indices.

        Outputs
        -------
        None
            Activates the selected section and deactivates every other section,
            retaining lazy calculation for inactive models.
        '''

        # Activate only the selected model and cancel queued work for inactive tabs
        for index, section in enumerate(sections):
            section.set_active(index == new)

    # Connect tab activation and create the document's shared introductory header
    tabs.on_change("active", active_changed)
    header = Div(
        text=(
            "<h1 style='margin:0 0 4px'>Interactive k–ω Dispersion Explorer</h1>"
            "<p style='margin:0;color:#475569'>Exact Python root solving, continuity "
            "tracking, and shared physical parameters. Every solution is shown together; "
            "live debounced numerical and slider edits update dependent models while global sheet "
            "matching keeps panel identities stable.</p>"
        ),
        sizing_mode="stretch_width",
        styles={"padding": "12px 8px 10px"},
    )
    # Attach the application root and optionally schedule the first lazy calculation
    doc.add_root(column(header, tabs, sizing_mode="stretch_width", name="application_root"))
    sections[0]._active = True
    if schedule_initial:
        sections[0].schedule_recompute(delay_ms=20)
    return sections
