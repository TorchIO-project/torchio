# Subject

The `Subject` is a data structure used to store
images associated with a subject and any other metadata necessary for
processing.

Subject objects can be sliced using the standard NumPy / PyTorch slicing
syntax, returning a new subject with sliced images.
This is only possible if all images in the subject have the same spatial
shape.

All transforms applied to a `Subject` are saved
in its `history` attribute.

::: torchio.Subject
