"""FedKit backend operations package.

Use this package from an interactive python shell:

```python
import fed_kit
response = fed_kit.upload("test.tflite", "test_model", [100, 200, 300], "test_type")
print(response)
print(response.text)
```
"""
from json import dumps

import requests

DEFAULT_URL = "http://localhost:8000/"


# Always change together with `UploadDataSerializer` in `train.serializers`.
def upload(
    tflite_file: str | None,
    coreml_file: str | None,
    name: str,
    tflite_layers: list[int] | None,
    coreml_layers: list[dict[str, str]] | None,
    data_type: str,
    base: str = DEFAULT_URL,
):
    """Upload model `file` and store it as `name` on the backend."""
    url = base + "/train/upload"
    data = {
        "name": name,
        "tflite_layers": tflite_layers,
        "coreml_layers": coreml_layers,
        "data_type": data_type,
    }
    files: dict = {"data": dumps(data, separators=(",", ":"))}
    if tflite_file is not None:
        assert tflite_layers is not None
        files["tflite"] = open(tflite_file, "rb")
    if coreml_file is not None:
        assert coreml_layers is not None
        files["coreml"] = open(coreml_file, "rb")
    return requests.post(url, files=files)
