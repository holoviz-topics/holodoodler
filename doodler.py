# Hack from https://github.com/facebook/prophet/issues/1889
import multiprocessing
multiprocessing.set_start_method("fork")

# Standard library imports
import configparser
import datetime
import json
import pathlib
import time
from typing import List, Optional, Tuple

# External dependencies imports
import holoviews as hv
import imageio
import numpy as np
import pandas as pd
import param
import panel as pn
import PIL
from PIL import ImageDraw

# Local imports
import doodler

# Load the bokeh extension for holoviews and panel
hv.extension('bokeh')

# ## Configuration

# Global holoviews parameters: no axis ticks and numbers on overlay plots
hv.opts.defaults(hv.opts.Overlay(xaxis='bare', yaxis='bare'))

# The class/color mapping is obtained from an INI file.

config = configparser.ConfigParser()
config.read('config.ini')
CLASS_COLOR_MAPPING = dict(config['classes'])

# ## Components
#
# ### Class toggle group
#
# The `ClassToggleGroup` component allows to toggle a class by clicking on its colorized button. The advantage of this component is that the buttons are layed out in a flexible container that wraps them line by line responsively.
#
# We first define an `Toggle` component that represents a button of the `ClassToggleGroup` component.


class Toggle(pn.reactive.ReactiveHTML):

    active = param.Boolean(False)

    klass = param.String()

    color = param.String()

    _template = """<button id="button" style="border-color:{{ color }};border-width:4px;border-radius:5%;padding-inline:10px;font-weight:{{ 'bold' if active else 'normal' }}" onclick="${_update}">{{ klass }}</button>"""

    _scripts = {
        'active': """
        if (data.active) {
            button.style.fontWeight = "bold"
        } else {
            button.style.fontWeight = "normal"
        }
        """
    }

    def _update(self, event):
        # One way update, a toggle can be only deactivated by setting .active to False programmatically.
        if not self.active:
            self.active = True


class ClassToggleGroup(pn.viewable.Viewer):

    active = param.String()

    class_color_mapping = param.Dict()

    def __init__(self, **params):
        super().__init__(**params)

        widgets = {}
        for i, (klass, color) in enumerate(self.class_color_mapping.items()):
            widget = Toggle(klass=klass, color=color)
            if i == 0:
                widget.active = True
            widget.param.watch(self._update_active, 'active')
            widgets[klass] = widget

        klass0 = next(iter(self.class_color_mapping))
        self.active = klass0
        self._widgets = widgets

    def _update_active(self, event):
        self._prev_active = self.active
        self._widgets[self._prev_active].active = False
        self.active = event.obj.klass

    def __panel__(self):
        # Add bottom margin to avoid the flexbox to overlap with a bottom widget.
        return pn.FlexBox(*self._widgets.values(), margin=(0, 0, 15, 0))


# ### DoodleDrawer
#
# The `DoodleDrawer` class provides the drawing functionality required for `Doodler`, i.e. the ability to quickly draw lines with different class/color and width. Its `doodles` property allows to obtain the lines drawn as a list of pandas dataframes.

class DoodleDrawer(pn.viewable.Viewer):

    # Required input

    class_color_mapping = param.Dict(precedence=-1)

    # Optional input

    class_toggle_group_type = param.ClassSelector(class_=ClassToggleGroup, is_instance=False)

    # UI elements

    line_width = param.Integer(default=2, bounds=(1, 10))

    clear_all = param.Event(label='Clear doodles')

    # Internal parameter

    class_toggle_group = param.Parameter(precedence=-1)

    label_class = param.Selector(precedence=-1)

    line_color = param.Selector(precedence=-1)

    def __init__(self, class_color_mapping, **params):
        self._accumulated_lines = []  # List of dataframes

        super().__init__(class_color_mapping=class_color_mapping, **params)

        classes = list(self.class_color_mapping)
        self.param.label_class.objects = classes
        self.param.label_class.default = self.label_class = classes[0]
        colors = list(self.class_color_mapping.values())
        self.param.line_color.objects = colors
        self.param.line_color.default = self.line_color = colors[0]

        if 'class_toggle_group_type' in params:
            self.class_toggle_group = self.class_toggle_group_type(class_color_mapping=self.class_color_mapping)
            def link(event):
                self.label_class = event.new
            self.class_toggle_group.param.watch(link, 'active')

        # Pipe used to initialize the draw plot and clear it in ._accumulate_drawn_lines()
        self._draw_pipe = hv.streams.Pipe(data=[])
        # The DynamicMap reacts to the parameters change to draw lines with the desired style.
        self._draw = hv.DynamicMap(self._clear_draw_cb, streams=[self._draw_pipe]).apply.opts(
            color=self.param.line_color, line_width=self.param.line_width
        ).opts(active_tools=['freehand_draw'])
        # Create a FreeHandDraw linked stream and attach it to the DynamicMap/
        # The DynamicMap plot is going to serve as a support for the draw tool,
        # and the data is going to be save in the stream (see .element or .data).
        self._draw_stream = hv.streams.FreehandDraw(source=self._draw)

        # This Pipe is going to send lines accumulated from previous drawing 'sessions',
        # a session including all the lines drawn between a parameter change (line_width, class, ...).
        self._drawn_pipe = hv.streams.Pipe()
        self._drawn = hv.DynamicMap(self._drawn_cb, streams=[self._drawn_pipe]).apply.opts(
            color='line_color', line_width='line_width'
        )

        # Set the ._accumulate_drawn_lines() callback on parameter changes to gather
        # the lines previously drawn.
        self.param.watch(self._accumulate_drawn_lines, ['line_color', 'line_width'])

        # Store the previous label class, this is used in ._accumulate_drawn_lines
        self._prev_label_class = self.label_class

    @param.depends('label_class', watch=True)
    def _update_color(self):
        self.line_color = self.class_color_mapping[self.label_class]

    def _clear_draw_cb(self, data: List):
        """Clear the lines drawn in a session.
        """
        # data is always []
        return hv.Contours(data)

    def _drawn_cb(self, data: Optional[List[pd.DataFrame]]):
        """Plot all the lines previously drawn.
        """
        return hv.Contours(data, kdims=['x', 'y'], vdims=['line_color', 'line_width'])

    def _accumulate_drawn_lines(self, event: Optional[param.parameterized.Event] = None):
        """Accumulate the drawn lines, clear the drawing plot and plot all
        the drawn lines.
        """
        # dframe() on a stream element that has multiple lines return a dataframe
        # with an empty line (filled with np.nan) separating the lines. To avoid
        # having to deal with that, .split() is used to obtain a dataframe per line.
        lines = [element.dframe() for element in self._draw_stream.element.split()]
        lines = [df_line for df_line in lines if not df_line.empty]
        if not lines:
            return
        # Add to each dataframe/line its properties and its label class
        for df_line in lines:
            for ppt in ['line_width', 'line_color']:
                if event:
                    df_line[ppt] = event.old if event.name == ppt else getattr(self, ppt)
                else:
                    # Ne event means that we want the current properties.
                    df_line[ppt] = getattr(self, ppt)
            df_line['label_class'] = self._prev_label_class
        self._accumulated_lines.extend(lines)
        # Clear the plot from the lines just drawn
        self._draw_pipe.event(data=[])
        # Clear the draw stream
        self._draw_stream.event(data={})
        # Plot all the lines drawn at this stage by sending them through this Pipe
        self._drawn_pipe.event(data=self._accumulated_lines)

        self._prev_label_class = self.label_class

    @param.depends('clear_all', watch=True)
    def _update_clear(self):
        self.clear()

    def clear(self):
        self._accumulated_lines = []
        self._draw_pipe.event(data=[])
        self._drawn_pipe.event(data=[])
        self._draw_stream.event(data={})

    @property
    def classes(self):
        return list(self.class_color_mapping.keys())

    @property
    def colormap(self):
        return list(self.class_color_mapping.values())

    @property
    def plot(self):
        return self._drawn * self._draw

    @property
    def doodles(self) -> List[pd.DataFrame]:
        if self._draw_stream.data:
            self._accumulate_drawn_lines()
        return self._accumulated_lines


# The geometry of the doodles obtained from `DoodleDrawer` is defined by a series of points referenced in a given coordinate system. What we want is to turn the doodles into a mask (each class being represented by an integer) whose dimension is equal to the dimension of the image the doodles will be associated with. The following functions allow to create such a mask from the doodles.

def _project_line_dimension(s: pd.Series, cur_range, target_range) -> pd.Series:
    assert ((cur_range[0] <= s) & (s <= cur_range[1])).all()
    assert cur_range[0] < cur_range[1]
    assert target_range[0] < target_range[1]
    return target_range[0] + (target_range[1] - target_range[0]) * (s - cur_range[0]) / (cur_range[1] - cur_range[0])


def project_doodles(
    doodles: List[pd.DataFrame],
    x_cur_range: Tuple[float, float],
    y_cur_range: Tuple[float, float],
    x_target_range: Tuple[int, int],
    y_target_range: Tuple[int, int],
) -> List[pd.DataFrame]:
    """
    Project and rescale the doodles from HoloViews to PIL
    """
    projected = []
    for df_doodle in doodles:
        df_proj = df_doodle.copy()

        df_proj['x_proj'] = _project_line_dimension(df_proj['x'], x_cur_range, x_target_range)
        df_proj['y_proj'] = _project_line_dimension(df_proj['y'], y_cur_range, y_target_range)

        # Because the origin is bottom left in bokeh and top left in PIL
        df_proj['y_proj'] = y_target_range[1] - df_proj['y_proj']
        projected.append(df_proj)
    return projected


def doodles_as_array(
    doodles: List[pd.DataFrame],
    img_width: int,
    img_height: int,
    colormap: List[str],
) -> np.ndarray:
    """
    Turn doodle lines into Numpy arrays. The line width is taken into account.
    """
    pimg = PIL.Image.new('L', (img_width, img_height), 0)
    drawing = ImageDraw.Draw(pimg)
    for doodle in doodles:
        # Project each line from the bokeh coordinate system to the one required to create them with PIL.
        # List of vertices (x, y)
        vertices = list(doodle[['x_proj', 'y_proj']].itertuples(index=False, name=None))
        # There's a unique width per line
        line_width = doodle.loc[0, 'line_width']
        # Index of the colomap + 1
        line_color = doodle.loc[0, 'line_color']
        fill_value = colormap.index(line_color) + 1
        drawing.line(
            vertices,
            width=line_width,
            fill=fill_value,
            joint='curve'
        )
    return np.array(pimg)


# ### Input image
#
# The `InputImage` component allows a user to select an image. An instance can be created with the `from_folder` class method that will find all the JPEG images in a folder. The `remove_img` method removes the current image from the list of images available and sets the next one, if available.

# +
# TODO: Where will we actually get the images from?

class InputImage(param.Parameterized):

    # UI elements

    location = param.Selector(label='Input image (.JPEG)')

    # Internal parameters

    width = param.Integer(default=600, precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._pane = pn.pane.HoloViews()
        self._load_image()

    @classmethod
    def from_folder(cls, imgs_folder, **params):
        jpegs = [
            p
            for p in pathlib.Path(imgs_folder).iterdir()
            if p.is_file() and p.suffix in ('.jpg', '.jpeg')
        ]
        jpegs = sorted(jpegs)
        input_image = cls(**params)
        input_image.param.location.objects = jpegs
        input_image.location = jpegs[0]
        return input_image

    @staticmethod
    def read_from_fs(path):
        img = PIL.Image.open(path)
        arr = np.array(img)
        # Some JPEG files have an alpha channel that we strip.
        arr = arr[:, :, :3]
        return arr

    @param.depends('location', watch=True)
    def _load_image(self):
        if not self.location:
            self.plot = hv.RGB(data=[]).opts(frame_width=self.width)
            self._pane.object = self.plot
            return
        self.array = array = self.read_from_fs(self.location)
        h, w, _ = array.shape
        # Preserve the aspect ratio
        self.plot = hv.RGB(array, bounds=(-1, -1, 1, 1)).opts(frame_width=self.width, aspect=w/h)
        self._pane.object = self.plot

    def remove_img(self):
        """
        Remove the current image and get the next one if available.
        """
        next_locations = self.param.location.objects[1:]
        self.param.location.objects = next_locations
        if next_locations:
            self.location = next_locations[0]
        else:
            self.location = None

    @property
    def pane(self):
        return self._pane


# ### Computation settings
#
# The `ComputationSettings` class declares all the parameters required by the algorithms perfoming the segmentation. UI-wise it provides the ability to switch to an *advanced* mode that displays more parameters to the user.

class ComputationSettings(pn.viewable.Viewer):

    # TODO: Are there any parameters that are meant to be displayed in the basic mode?

    advanced = param.Boolean(default=False)

    # Post-processing settings

    crf_theta = param.Number(default=1, bounds=(1, 100), step=1, label="Blur factor", precedence=11)

    crf_mu = param.Number(default=1, bounds=(1, 100), step=1, label="Model independence factor", precedence=1)

    crf_downsample_factor = param.Integer(default=2, bounds=(1, 6), label="CRF downsample factor", precedence=1)

    gt_prob = param.Number(default=0.9, bounds=(0.5, 0.99), step=0.1, label="Probability of doodle", precedence=1)

    # Classifier settings

    rf_downsample_value = param.Integer(default=1, bounds=(1, 20), step=1, label="Classifier downsample factor", precedence=1)

    n_sigmas = param.Integer(default=2, bounds=(2, 6), label="Number of scales", precedence=1)

    # Fixed parameters (hard-coded in Dash doodler)

    multichannel = param.Boolean(True, constant=True, precedence=-1)

    intensity = param.Boolean(True, constant=True, precedence=-1)

    edges = param.Boolean(True, constant=True, precedence=-1)

    texture = param.Boolean(True, constant=True, precedence=-1)

    sigma_min = param.Integer(1, constant=True, precedence=-1)

    sigma_max = param.Integer(16, constant=True, precedence=-1)

    # Precedence thresholds

    _ADVANCED = 0
    _BASIC = 10

    def __init__(self, **params):
        super().__init__(**params)
        self._pane = pn.Param(self.param, display_threshold=self._BASIC, sizing_mode='stretch_width')

    @param.depends('advanced', watch=True)
    def _update_threshold(self):
        self._pane.display_threshold = self._ADVANCED if self.advanced else self._BASIC

    def as_dict(self):
        return {
            p: v
            for p, v in self.param.values().items()
            if p not in ('name', 'advanced')
        }

    def __panel__(self):
        return self._pane


class Info(pn.viewable.Viewer):

    def __init__(self):
        # print(params)
        super().__init__()
        self._pane = pn.pane.Alert(min_height=150, sizing_mode='stretch_both')

    def update(self, msg, msg_type='primary'):
        self._pane.object = msg
        self._pane.alert_type = msg_type

    def add(self, msg):
        self._pane.object += f'<br>{msg}'

    def reset(self):
        self._pane.object = ''
        self._pane.alert_type = 'primary'

    def __panel__(self):
        return self._pane


# ## Combining the components with the segmentation computation
#
# The `ApplicationBase` class uses and combines the components introduced above with components and methods dedicated to the segmentation itself, that call the learning algorithms.

class ApplicationBase(param.Parameterized):

    # Main components

    settings = param.ClassSelector(class_=ComputationSettings, is_instance=True)

    doodle_drawer = param.ClassSelector(class_=DoodleDrawer, is_instance=True)

    input_image = param.ClassSelector(class_=InputImage, is_instance=True)

    info = param.ClassSelector(class_=Info, is_instance=True)

    # Segmentation UI

    compute_segmentation = param.Event(label='Compute segmentation')

    clear_segmentation = param.Event(label='Clear segmentation')

    save_segmentation = param.Event(label='Save segmentation and continue')

    # Customizable HoloViews styles (hidden from the GUI, settable in the constructor)

    canvas_width = param.Integer(default=600)

    def __init__(self, **params):
        super().__init__(**params)
        self._img_pane = pn.pane.HoloViews()
        self._init_img_pane()
        self._init_segmentation_output()

    def _init_img_pane(self):
        self._img_pane.object = self.input_image.plot * self.doodle_drawer.plot

    @param.depends('input_image.location', watch=True)
    def _reset(self):
        # Selecting a new image so reset/clear the app.
        self.doodle_drawer.clear()
        self._clear_segmentation()
        self.info.reset()

    def _init_segmentation_output(self):
        self._segmentation_color = None
        self._segmentation = None
        self._mask_doodles = None

    @param.depends('clear_segmentation', watch=True)
    def _clear_segmentation(self):
        self._init_img_pane()
        self._init_segmentation_output()

    @param.depends('compute_segmentation', watch=True)
    def _compute_segmentation(self):
        doodles = self.doodle_drawer.doodles
        if not doodles:
            self.info.update('Draw doodles before trying to run the algorithm.', 'danger')
            return
        if not self.input_image.location:
            self.info.update('Input image not loaded.', 'danger')
            return

        with pn.param.set_values(self._img_pane, loading=True):
            start_time = time.time()
            self.info.update('Start...')

            self.info.add('Projecting/Converting doodles into a mask...')
            img_height, img_width, _ = self.input_image.array.shape
            projected_doodles = project_doodles(
                doodles,
                x_cur_range=self.input_image.plot.range('x'),
                y_cur_range=self.input_image.plot.range('y'),
                x_target_range=(0, img_width),
                y_target_range=(0, img_height),
            )
            # Get a mask with the doodles
            self._mask_doodles = doodles_as_array(
                projected_doodles,
                img_width=img_width,
                img_height=img_height,
                colormap=self.doodle_drawer.colormap,
            )

            # Long computation...
            self.info.add('Core segmentation computation...')
            self._segmentation = doodler.segmentation(
                img=self.input_image.array,
                mask=self._mask_doodles,
                **self.settings.as_dict(),
            )

            self.info.add('Colorizing the segmentation...')
            self._segmentation_color = doodler.label_to_colors(
                self._segmentation,
                self.input_image.array[:, :, 0] == 0,
                colormap=self.doodle_drawer.colormap,
                color_class_offset=-1,
            )

            self.info.add('Rendering the results...')
            hv_segmentation_color = hv.RGB(self._segmentation_color, bounds=(-1, -1, 1, 1)).opts(alpha=0.5)
            self._img_pane.object = self._img_pane.object * hv_segmentation_color
            duration = round(time.time() - start_time, 1)
            self.info.add(f'Process done in {duration}s.')

    @param.depends('save_segmentation', watch=True)
    def _save_segmentation(self):
        """
        TODO: Define what do save, how and where.
        """
        if self._segmentation is None:
            self.info.update('Run first a segmentation before saving.', 'danger')
            return

        self.info.update('Saving results...', 'success')
        root_res_dir = pathlib.Path('results')
        root_res_dir.mkdir(exist_ok=True)

        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        res_dir = root_res_dir / now
        res_dir.mkdir()

        input_img_file = res_dir / 'input.jpeg'
        imageio.imwrite(input_img_file, self.input_image.array)
        doodles_file = res_dir / 'doodles.jpeg'
        imageio.imwrite(doodles_file, self._mask_doodles)
        col_seg_file = res_dir / 'colorized_segmentation.png'
        imageio.imwrite(col_seg_file, self._segmentation_color)

        content = {}
        content['time'] = now
        content['user'] = 'placeholder'
        content['settings'] = self.settings.as_dict()
        content['classes'] = self.doodle_drawer.classes
        content['colormap'] = self.doodle_drawer.colormap
        in_ = {}
        in_['image'] = str(input_img_file)
        content['input'] = in_
        out = {}
        out['doodles'] = str(doodles_file)
        out['colorized_segmentation'] = str(col_seg_file)
        content['output'] = out

        json_file = res_dir / 'info.json'
        with open(json_file, 'w') as finfo:
            json.dump(content, finfo, indent=4)
        self.info.add('Done! Onto the next one!')

    @param.depends('_save_segmentation', watch=True)
    def _remove_image(self):
        self.input_image.remove_img()

    @property
    def plot_pane(self):
        return self._img_pane


# The different components are instantiated and passed to `ApplicationBase`.

doodle_drawer = DoodleDrawer(class_color_mapping=CLASS_COLOR_MAPPING, class_toggle_group_type=ClassToggleGroup)
input_image = InputImage.from_folder('examples/images')
settings = ComputationSettings(name='Post-processing/Classifier settings')
info = Info()
appb = ApplicationBase(settings=settings, doodle_drawer=doodle_drawer, info=info, input_image=input_image)

# ## Layout
#
# ### Notebook application
#
# First a simple application is put together in the notebook by laying out the components in `Row` and `Column` Panel panes. This step is very useful when developing the application.

side_bar = pn.Column(
    appb.input_image.param.location,
    appb.input_image.param.width,
    pn.pane.HTML('<b>Doodling options</b>'),
    appb.doodle_drawer.class_toggle_group,
    appb.doodle_drawer.param.line_width,
    appb.doodle_drawer.param.clear_all,
    appb.settings,
    pn.widgets.Button.from_param(appb.param.compute_segmentation, button_type='primary'),
    pn.widgets.Button.from_param(appb.param.clear_segmentation, button_type='warning'),
    pn.widgets.Button.from_param(appb.param.save_segmentation, button_type='success'),
    appb.info,
    width=350,  # Width set to avoid issues with the class FlexBox. Slightly less than the side_bar width of the Material template (370).
)
main = appb.plot_pane

# ## Deployable application
#
# While the notebook application already provides all the functionnality we require, its design should be improved a little to make it a proper web app. We're embedding it into one of the templates provided by Panel and add a few elements like the USGS logo. Serve the application by running `panel serve doodler.ipynb --show`.

template = pn.template.MaterialTemplate(
    title='Doodler',
    logo='assets/1280px-USGS_logo.png',
    header_background='#000000',
    sidebar=[side_bar],
    main=[main],
)

template.servable()
