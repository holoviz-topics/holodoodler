import configparser
import logging
import logging.config

import panel as pn

# Import the components
from doodler.components import (
    Application,
    ClassToggleGroup,
    ComputationSettings,
    DoodleDrawer,
    Info,
    InputImage,
)

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s\t%(levelname)s\t%(module)s:%(lineno)d\t%(message)s',
            'datefmt': '%m/%d/%Y %I:%M:%S %p'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'app.log',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
logger = logging.getLogger(__name__)
logger.debug('Start app execution.')

# Read the config file
config = configparser.ConfigParser()
config.read('config.ini')
CLASS_COLOR_MAPPING = dict(config['classes'])

# Instantiate  the main components required by the Application
doodle_drawer = DoodleDrawer(class_color_mapping=CLASS_COLOR_MAPPING, class_toggle_group_type=ClassToggleGroup)
input_image = InputImage.from_folder('examples/images')
settings = ComputationSettings(name='Post-processing/Classifier settings')
info = Info()

# Create the application
app = Application(settings=settings, doodle_drawer=doodle_drawer, info=info, input_image=input_image)

# Layout the components
side_bar = [
    app.input_image.param.location,
    pn.pane.HTML('<b>Doodling options</b>'),
    app.doodle_drawer.class_toggle_group,
    app.doodle_drawer.param.line_width,
    app.doodle_drawer.param.clear_all,
    app.settings,
    pn.widgets.Button.from_param(app.param.compute_segmentation, button_type='primary'),
    pn.widgets.Button.from_param(app.param.clear_segmentation, button_type='warning'),
    pn.widgets.Button.from_param(app.param.save_segmentation, button_type='success'),
    app.info,
]
main = app.plot_pane

# Populate the template with the side bar layout and the main layout
template = pn.template.MaterialTemplate(
    title='Doodler',
    logo='assets/1280px-USGS_logo.png',
    header_background='#000000',
    sidebar=side_bar,
    main=[main],
)

# Launch the app (`panel serve doodler.py``)
logger.debug('Serving the Panel app...')
template.servable()
