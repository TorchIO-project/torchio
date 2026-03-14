# 3D Slicer GUI

[3D Slicer](https://www.slicer.org/) is an open-source software platform for
medical image informatics, image processing,
and three-dimensional visualization.

TorchIO provides a 3D Slicer extension for quick experimentation and
visualization of the package features without any coding.

The TorchIO extension can be easily installed using the
[Extensions Manager](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html).

The code and installation instructions are available on
[GitHub](https://github.com/fepegar/SlicerTorchIO).

!!! note
    The Preview version (built nightly) is recommended. You can download
    and install Slicer from [their download website](https://download.slicer.org/)
    or, if you are on macOS, using [Homebrew](https://docs.brew.sh/):

    ```
    brew tap homebrew/cask-versions && brew cask install slicer-preview
    ```

## TorchIO Transforms

This module can be used to quickly visualize the effect of each transform
parameter.
That way, users can have an intuitive feeling of what the output
of a transform looks like without any coding at all.

![TorchIO Transforms module for 3D Slicer](https://raw.githubusercontent.com/fepegar/SlicerTorchIO/master/Screenshots/TorchIO.png)

### Usage example

Go to the `Sample Data` module to get an image we can use:

![Go to Sample Data module](https://raw.githubusercontent.com/fepegar/SlicerTorchIO/master/Screenshots/usage_1.png)

Click on an image to download, for example MRHead[^1],
and go to the `TorchIO Transforms` module:

[^1]: All the data in `Sample Data` can be downloaded and used in the TorchIO
    Python library using the `torchio.datasets.slicer.Slicer` class.

![Download MRHead and go to TorchIO Transforms module](https://raw.githubusercontent.com/fepegar/SlicerTorchIO/master/Screenshots/usage_2.png)

Select the input and output volume nodes:

![Select volume nodes](https://raw.githubusercontent.com/fepegar/SlicerTorchIO/master/Screenshots/usage_3.png)

Modify the transform parameters and click on `Apply transform`.
Hovering the mouse over the transforms will show tooltips extracted from the
TorchIO documentation.

![Apply transform](https://raw.githubusercontent.com/fepegar/SlicerTorchIO/master/Screenshots/usage_4.png)

You can click on the `Toggle volumes` button to switch between input and
output volumes.
