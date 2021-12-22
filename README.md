# Doodler

This repository hosts `Doodler`, a Python application (TBD) that allows interactive construction of sparse labeling for image segmentation using deep neural networks, based on the [Holoviz](https://holoviz.org/) ecosystem.

## Project background

Deep neural networks have proven to be powerful tools for automatic classification of images (e.g. identifying regions of sand, water and vegetation from drone-collected beach images).  These techniques are used extensively by USGS researchers to understand how beach and coastal environments are impacted by storms and other events. The first step in using this technique is sparsely annotating a collection of images by hand to train the models.

The USGS currently has an open source package - [dash-doodler](https://github.com/dbuscombe-usgs/dash_doodler) - that performs the task, but does not run in multi-user JupyterHub environments. The USGS needs a version that is fully usable from Jupyter Notebooks so that researchers, interns and volunteers can effectively start working immediately in a JupyterHub environment.

The USGS seeks to achieve this goal by using the HoloViz/Panel ecosystem, which will integrate into their existing deployed JupyterHub environment.  
