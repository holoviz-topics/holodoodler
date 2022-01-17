import logging

from .components import (
    Application,
    ClassToggleGroup,
    ComputationSettings,
    DoodleDrawer,
    Info,
    InputImage,
    Toggle,
)

__all__ = [
    "Application",
    "ClassToggleGroup",
    "ComputationSettings",
    "DoodleDrawer",
    "Info",
    "InputImage",
    "Toggle",
]

# Set default logging handler for a library
logging.getLogger(__name__).addHandler(logging.NullHandler())
