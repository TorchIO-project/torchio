# Installation

=== "uv"

    ```
    uv add torchio
    ```

=== "pip"

    ```
    pip install torchio
    ```

## Optional extras

For NIfTI-Zarr support (chunked, lazy-loadable volumes):

=== "uv"

    ```
    uv add torchio --extra zarr
    ```

=== "pip"

    ```
    pip install "torchio[zarr]"
    ```

For cloud storage (HTTP/HTTPS URLs work out of the box):

=== "Azure Blob"

    ```
    pip install "torchio[azure]"
    ```

=== "S3"

    ```
    pip install "torchio[s3]"
    ```

=== "Google Cloud"

    ```
    pip install "torchio[gcs]"
    ```

For an interactive 3D viewer in Jupyter
([NiiVue](https://niivue.com/)):

```
pip install "torchio[niivue]"
```

## Verify

```python
import torchio as tio

print(tio.__version__)
```
