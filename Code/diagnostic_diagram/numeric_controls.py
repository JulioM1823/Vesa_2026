'''
Purpose
-------
Pair Bokeh sliders with exact numerical text inputs and synchronize
physical parameters across the model sections of one document.

Inputs
------
None

Outputs
-------
validate_parameter: callable
    Physical-domain validation independent of suggested slider ranges.
ExactSlider, SharedParameterState: type
    Exact-value controls and document-local parameter synchronization.
'''
# Import escaping, immutable-definition updates, and numerical validation
from html import escape
from dataclasses import replace
import math

# Import the layout and widget types used by the paired controls
from bokeh.layouts import column, row
from bokeh import events
from bokeh.models import Div, RangeSlider, TextInput


def validate_parameter(key, value):
    '''
    Purpose
    -------
    Reject nonfinite values and values outside the physical domain assigned
    to a recognized parameter key.

    Inputs
    ------
    key: str
        Parameter identifier. Angle keys use degrees; other physical units
        follow the model parameter definitions.
    value: float
        Numerical value to validate independently of a slider display range.

    Outputs
    -------
    None
        Raise ValueError for a violated domain. Unrecognized finite keys pass
        without an additional domain constraint.
    '''
    # Reject undefined values before applying key-specific physical domains
    if not math.isfinite(value):
        raise ValueError('Enter a finite number.')
    # Keep inclination and azimuth within the supported degree interval
    if key in {'theta_deg', 'phi_deg'} and not -180 <= value <= 180:
        raise ValueError('Angle must lie between −180 and 180 degrees.')
    # Enforce strict positivity for speeds, scales, and relaxation times
    if key in {'cs', 'g', 'H', 'tau', 'width_k', 'width_f'} and value <= 0:
        raise ValueError('Value must be greater than zero.')
    # Allow zero only for parameters whose physical domains include it
    if key in {'a', 'dz', 'wac', 'central_k', 'central_f', 'strength', 'contour_sigma'} and value < 0:
        raise ValueError('Value must be nonnegative.')
    # Exclude the isothermal and sub-unity specific-heat-ratio values
    if key == 'gamma' and value <= 1:
        raise ValueError('Specific-heat ratio must exceed one.')


class ExactSlider:
    '''
    Purpose
    -------
    Keep a Bokeh slider as the accepted value and pair it with validated
    text input that commits on Enter or blur.

    Inputs
    ------
    slider: bokeh.models.Slider or RangeSlider
        Existing scalar or interval slider to synchronize.
    key: str
        Parameter key used for validation and widget naming.
    validator: callable, optional
        Scalar validation function; defaults to validate_parameter(key, value).
    hard_bounds: bool, optional
        Reject values outside the current slider bounds when True.
    integer: bool, optional
        Require integer-valued input when True.

    Outputs
    -------
    ExactSlider
        Wrapper exposing slider, input, label, message, and layout attributes.

    Notes
    -----
    Invalid or incomplete edits retain the accepted slider value. Valid
    values outside suggested bounds expand those bounds unless hard_bounds
    is True. This class registers slider, input, and submit callbacks.
    '''
    def __init__(self, slider, key, *, validator=None, hard_bounds=False, integer=False):
        '''
        Purpose
        -------
        Create the exact-value widgets and register their synchronization
        callbacks.

        Inputs
        ------
        slider: bokeh.models.Slider or RangeSlider
            Slider whose value remains the accepted source of state.
        key: str
            Physical parameter key or other validated control identifier.
        validator: callable, optional
            Scalar validator raising ValueError for unacceptable values.
        hard_bounds, integer: bool, optional
            Enforce slider limits and integer-valued input, respectively.

        Outputs
        -------
        None
            Store widgets on this instance, hide the slider title/value display,
            and register callbacks on the supplied slider and paired text input.
        '''

        # Store validation rules and initialize the callback recursion guard
        self.slider, self.key = slider, key
        self.validator = validator or (lambda value: validate_parameter(key, value))
        self.hard_bounds, self.integer = hard_bounds, integer
        self._syncing = False
        # Move the slider title into a wrapping label above the paired widgets
        self.label = Div(text=escape(slider.title), visible=bool(slider.title), sizing_mode='stretch_width',
                         styles={'font-size':'12px', 'overflow-wrap':'anywhere'})
        slider.title = ''
        slider.show_value = False
        # Create the exact text input and an initially hidden validation message
        self.input = TextInput(value=self.format(slider.value), title='', placeholder='Exact value',
                               width=100, name=(slider.name or key)+':exact')
        self.message = Div(text='', visible=False, sizing_mode='stretch_width',
                           styles={'font-size': '11px', 'color': '#b91c1c'})
        # Arrange the slider and text input together with their feedback
        self.layout = column(self.label, row(slider, self.input, sizing_mode='stretch_width'),
                             self.message, sizing_mode='stretch_width', spacing=0)
        # Synchronize value and disabled-state changes in both directions
        slider.on_change('value', self._from_slider)
        slider.on_change('disabled', self._disabled)
        self.input.disabled = slider.disabled
        self.input.on_change('value', self._from_input)
        # Current Bokeh emits ValueSubmit on Enter without committing value.
        if hasattr(events, 'ValueSubmit'):
            self.input.on_event(events.ValueSubmit, self._submitted)

    def _submitted(self, event):
        '''
        Purpose
        -------
        Commit the text supplied by a Bokeh ValueSubmit event.

        Inputs
        ------
        event: bokeh.events.ValueSubmit
            Submit event containing the entered value as text.

        Outputs
        -------
        None
            Update the TextInput value when it differs, triggering validation.
        '''

        # Let the ordinary text-change callback validate the submitted value
        if self.input.value != event.value:
            self.input.value = event.value

    @staticmethod
    def format(value):
        '''
        Purpose
        -------
        Represent accepted scalar or interval values as exact float text.

        Inputs
        ------
        value: float, tuple, or list
            Scalar slider value or the two bounds of a range slider.

        Outputs
        -------
        text: str
            Float repr text, or comma-separated float repr text for a range.
        '''

        # Preserve both range endpoints using round-trippable float text
        if isinstance(value, (tuple, list)):
            return ', '.join(repr(float(v)) for v in value)
        return repr(float(value))

    def _disabled(self, attr, old, new):
        '''
        Purpose
        -------
        Mirror the slider disabled state onto its paired text input.

        Inputs
        ------
        attr: str
            Changed Bokeh property name; accepted for the callback signature.
        old, new: object
            Previous and current property values.

        Outputs
        -------
        None
            Assign new to the TextInput disabled property.
        '''

        # Apply the same editability to both controls
        self.input.disabled = new

    def _from_slider(self, attr, old, new):
        '''
        Purpose
        -------
        Refresh the exact-value text and clear validation feedback after an
        accepted slider change.

        Inputs
        ------
        attr: str
            Changed Bokeh property name; accepted for the callback signature.
        old, new: object
            Previous and current property values.

        Outputs
        -------
        None
            Update the TextInput and validation message unless a synchronization
            callback is already active.
        '''

        if self._syncing:
            return
        # Hold the recursion guard until the accepted value has been displayed
        self._syncing = True
        try:
            self.input.value = self.format(new)
            self.message.visible = False
        finally:
            self._syncing = False

    def _from_input(self, attr, old, new):
        '''
        Purpose
        -------
        Parse and validate exact-value text before updating the slider.

        Inputs
        ------
        attr: str
            Changed Bokeh property name; accepted for the callback signature.
        old: str
            Previous input text; accepted for the callback signature.
        new: str
            Scalar text or two comma-separated range bounds.

        Outputs
        -------
        None
            Apply valid values, expanding non-hard slider bounds as needed.
            Display escaped error text and retain the accepted value on failure.

        Notes
        -----
        Integer and hard-bound checks are applied to each parsed value. Range
        bounds must be ordered. A guard prevents recursive callbacks, and a
        finally block restores it even if a widget assignment fails.
        '''

        if self._syncing:
            return
        try:
            # Parse either one scalar or exactly two range endpoints
            parts = new.split(',') if isinstance(self.slider, RangeSlider) else [new]
            if len(parts) != (2 if isinstance(self.slider, RangeSlider) else 1):
                raise ValueError('Enter two bounds separated by a comma.')
            values = [float(part.strip()) for part in parts]
            # Check physical, integer, and hard-bound constraints before mutation
            for value in values:
                self.validator(value)
                if self.integer and not value.is_integer():
                    raise ValueError('Enter an integer.')
                if self.hard_bounds and not self.slider.start <= value <= self.slider.end:
                    raise ValueError(f'Value must lie in [{self.slider.start}, {self.slider.end}].')
            # Reject reversed intervals before assigning the RangeSlider value
            if len(values) == 2 and values[0] > values[1]:
                raise ValueError('Lower bound must not exceed upper bound.')
        # Keep the previous slider value when parsing or validation fails
        except (TypeError, ValueError, OverflowError) as exc:
            self.message.text = escape(str(exc))
            self.message.visible = True
            return
        self._syncing = True
        try:
            # Expand only the suggested display range to contain accepted values
            expanded = min(values) < self.slider.start or max(values) > self.slider.end
            if expanded:
                self.slider.start = min(self.slider.start, *values)
                self.slider.end = max(self.slider.end, *values)
            # Commit the accepted scalar or interval and report range expansion
            self.slider.value = tuple(values) if len(values) == 2 else values[0]
            self.message.text = 'Slider range expanded to include the exact value.' if expanded else ''
            self.message.visible = expanded
            self.input.value = self.format(self.slider.value)
        finally:
            self._syncing = False


# Define the physical units required for document-wide parameter sharing
SHARED_PARAMETER_UNITS = {
    'a': 'km s⁻¹', 'cs': 'km s⁻¹', 'N2': 's⁻²', 'wac': 'rad s⁻¹',
    'theta_deg': 'deg', 'phi_deg': 'deg', 'dz': 'km', 'g': 'km s⁻²',
    'H': 'km', 'gamma': 'dimensionless', 'tau': 's',
}


class SharedParameterState:
    '''
    Purpose
    -------
    Maintain one set of compatible physical parameter values for all model
    sections in a Bokeh document.

    Inputs
    ------
    sections: sequence
        Model sections exposing widgets, definition, callback suppression,
        and parameter-change handlers.

    Outputs
    -------
    SharedParameterState
        Shared values and synchronization callbacks attached to the sections.

    Notes
    -----
    Only keys listed in SHARED_PARAMETER_UNITS are discovered for sharing.
    Their declared units must agree. Existing section values are normalized
    to the first discovered value for each key, which also becomes the shared
    reset default. Inactive model sections retain their lazy computation.
    '''
    def __init__(self, sections):
        '''
        Purpose
        -------
        Collect compatible shared controls, synchronize their initial values,
        and install document-wide reset defaults.

        Inputs
        ------
        sections: sequence
            Model sections whose compatible widgets participate in shared state.

        Outputs
        -------
        None
            Attach this state to sections, register callbacks, and replace their
            immutable parameter definitions with synchronized default values.
            Raise ValueError if a shared parameter declares incompatible units.
        '''

        # Initialize one parameter store and a guard against recursive updates
        self.sections = sections
        self.values = {}
        self._updating = False
        # Discover shared widgets and reject unit mismatches before linking them
        for section in sections:
            section.parameter_state = self
            for key, widget in section.widgets.items():
                if key not in SHARED_PARAMETER_UNITS:
                    continue
                item = next(item for item in section.definition.parameters if item.key == key)
                if item.unit != SHARED_PARAMETER_UNITS[key]:
                    raise ValueError(f'Cannot share {key}: incompatible physical units {item.unit}.')
                self.values.setdefault(key, float(widget.value))
                widget.on_change('value', self._callback(section, key))
        # Synchronize initial values without requesting solver recalculations
        self.apply(self.values, notify=False)
        # Record the synchronized values as reset defaults for every section
        for section in sections:
            section.definition = replace(section.definition, parameters=tuple(
                replace(item, default=self.values[item.key]) if item.key in self.values else item
                for item in section.definition.parameters))

    def _callback(self, section, key):
        '''
        Purpose
        -------
        Create a Bokeh change handler bound to a section and parameter key.

        Inputs
        ------
        section: object
            Section that originates the value change.
        key: str
            Shared parameter identifier captured by the callback.

        Outputs
        -------
        changed: callable
            Bokeh property callback that applies accepted values across sections.
        '''

        def changed(attr, old, new):
            '''
            Purpose
            -------
            Propagate a new shared control value when callbacks are not suppressed.

            Inputs
            ------
            attr: str
                Changed Bokeh property name; accepted for the callback signature.
            old: object
                Previous property value; accepted for the callback signature.
            new: float
                Updated scalar control value.

            Outputs
            -------
            None
                Apply the value with the captured section as its origin, unless a
                shared update or section-level suppression is active.
            '''

            # Propagate only direct user changes outside a synchronization pass
            if not self._updating and not section._suppress_callbacks:
                self.apply({key: float(new)}, origin=section)
        return changed

    def apply(self, values, *, origin=None, notify=True):
        '''
        Purpose
        -------
        Validate supplied values, synchronize matching widgets, and optionally
        notify sections whose controls changed.

        Inputs
        ------
        values: mapping of str to float
            Parameter updates in the units declared by the model definitions.
        origin: object, optional
            Initiating section, excluded from the additional change notifications.
        notify: bool, optional
            Invoke one parameter-change handler per affected section when True.

        Outputs
        -------
        None
            Update shared state, widget ranges and values, and section callbacks.
            Raise ValueError before applying values that fail physical validation.

        Notes
        -----
        Widget callbacks are suppressed while controls are synchronized. Each
        affected section receives the dispersion role if any changed parameter
        has that role; otherwise it receives the diagram role. Guards and prior
        section suppression flags are restored in finally blocks.
        '''

        # Validate all updates before altering shared values or widgets
        for key, value in values.items():
            validate_parameter(key, float(value))
        self._updating = True
        affected = []
        try:
            # Apply matching widget changes while preserving callback suppression
            self.values.update(values)
            for section in self.sections:
                changed = []
                previous_suppression = section._suppress_callbacks
                section._suppress_callbacks = True
                try:
                    for item in section.definition.parameters:
                        if item.key not in values:
                            continue
                        widget = section.widgets[item.key]
                        value = float(values[item.key])
                        # Keep each display range wide enough for the shared value
                        if widget.value != value:
                            widget.start = min(widget.start, value)
                            widget.end = max(widget.end, value)
                            widget.value = value
                            changed.append(item)
                finally:
                    section._suppress_callbacks = previous_suppression
                # Collect non-origin sections that require an explicit notification
                if changed and section is not origin:
                    affected.append((section, changed))
            # Coalesce each section update into one callback with the strongest role
            if notify:
                for section, changed in affected:
                    role = 'dispersion' if any(item.role == 'dispersion' for item in changed) else 'diagram'
                    section._parameter_changed(changed[0].key, role, 'value', None, None)
        finally:
            self._updating = False

    def reset(self, section):
        '''
        Purpose
        -------
        Apply a section's stored default values to the shared controls.

        Inputs
        ------
        section: object
            Section whose definition.defaults mapping supplies the reset values.

        Outputs
        -------
        None
            Synchronize defaults across matching controls and notify other
            affected sections through apply().
        '''

        # Reuse normal validation and synchronization for the stored defaults
        self.apply(dict(section.definition.defaults), origin=section)
