import math
import importlib
import functools

resnet_batch_inputs = [[[1, 64, 64, 64], [64], [64], [64], [64], [1, 64, 64, 64]], [[1, 64, 64, 64], [64], [64], [64], [64], [1, 64, 64, 64]], [[1, 128, 32, 32], [128], [128], [128], [128], [1, 128, 32, 32]], [[1, 128, 32, 32], [128], [128], [128], [128], [1, 128, 32, 32]], [[1, 128, 32, 32], [128], [128], [128], [128], [1, 128, 32, 32]], [[1, 256, 16, 16], [256], [256], [256], [256], [1, 256, 16, 16]], [[1, 256, 16, 16], [256], [256], [256], [256], [1, 256, 16, 16]], [[1, 256, 16, 16], [256], [256], [256], [256], [1, 256, 16, 16]], [[1, 512, 8, 8], [512], [512], [512], [512], [1, 512, 8, 8]], [[1, 512, 8, 8], [512], [512], [512], [512], [1, 512, 8, 8]], [[1, 512, 8, 8], [512], [512], [512], [512], [1, 512, 8, 8]]]
tmp = resnet_batch_inputs[0]
resnet_batch_inputs[0] = resnet_batch_inputs[2]
resnet_batch_inputs[2] = tmp

def generate_inputs(size):
    import numpy as np

    np.random.seed(17)
    inputs = resnet_batch_inputs[size]
    inputs_batch_modified = inputs
    inputs_batch_modified = []
    for inp in inputs:
        if inp[0] == 1:
            inputs_batch_modified.append([32] + inp[1:])
        else:
            inputs_batch_modified.append(inp)

    inputs = [np.random.uniform(0, 1000, size=shape) for shape in inputs_batch_modified]
    return inputs


def try_import(backend):
    try:
        return importlib.import_module(f".fusion_{backend}", __name__)
    except ImportError:
        return None


def get_callable(backend, size, device="cpu"):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, "prepare_inputs"):
        inputs = backend_module.prepare_inputs(inputs, device=device)
    return functools.partial(backend_module.run, inputs, device=device)

__implementations__ = (
    "jax",
    "numpy",
    "pytorch",
    "tensorflow",
)
