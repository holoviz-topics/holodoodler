# Doodler

This repository hosts `Doodler`, a Python application that allows interactive construction of sparse labeling for image segmentation using deep neural networks.

## Project background

Deep neural networks have proven to be powerful tools for automatic classification of images (e.g. identifying regions of sand, water and vegetation from drone-collected beach images).  These techniques are used extensively by USGS researchers to understand how beach and coastal environments are impacted by storms and other events. The first step in using this technique is sparsely annotating a collection of images by hand to train the models.

The USGS currently has an open source package - [dash-doodler](https://github.com/dbuscombe-usgs/dash_doodler) - that performs the task, but does not run in multi-user JupyterHub environments. The USGS needs a version that is fully usable from Jupyter Notebooks so that researchers, interns and volunteers can effectively start working immediately in a JupyterHub environment.

The USGS seeks to achieve this goal by using the HoloViz/Panel ecosystem, which will integrate into their existing deployed JupyterHub environment.  

## Installation

Run the following command to create a *conda* environment named `doodler-dev`:

```
conda env create --file environment_dev.yaml
```

## Development

The application currently lives in the Jupyter notebook `doodler.ipynb`. The development process of such an app can be observed in the notebook where (1) a series of small and well defined components are built, (2) then they're put together in a class that links them when required, (3) a simple layout is built and displayed in the notebook for testing purposes, and (4) a more polished app is built based on a Panel template and is made deployable.

The notebook can be opened with:

```
jupyter lab doodler.ipynb
```

The application can be launched with:

```
panel serve doodler.ipynb --show
```

## Video

This video shows the notebook content, how to use the application and the state of the application when deployed.


https://user-images.githubusercontent.com/35924738/147021690-5e525c4c-326a-491b-9d27-994250ff9fbb.mp4

