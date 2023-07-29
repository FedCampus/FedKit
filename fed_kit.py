"""FedKit backend operations package.

Use this package from an interactive python shell:

```python
import fed_kit
response = fed_kit.upload("test.tflite", "test_model", [100, 200, 300], "test_type")
print(response)
print(response.text)
```
"""
import requests

DEFAULT_URL = "http://localhost:8000/"


# Always change together with `UploadDataSerializer` in `train.serializers`.
def upload(
    file: str,
    name: str,
    layers_sizes: list[int],
    data_type: str,
    base: str = DEFAULT_URL,
):
    """Upload model `file` and store it as `name` on the backend."""
    url = base + "/train/upload"
    files = {"file": open(file, "rb")}
    data = {
        "name": name,
        "layers_sizes": layers_sizes,
        "data_type": data_type,
    }
    return requests.post(url, data=data, files=files)
