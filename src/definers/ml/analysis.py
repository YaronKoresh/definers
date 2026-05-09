from __future__ import annotations

import inspect
import warnings
from collections import Counter, OrderedDict

from definers.runtime_numpy import get_array_module

np = get_array_module()


def kmeans_k_suggestions(X, k_range=range(2, 20), random_state=None):
    from definers.system.download_activity import create_activity_reporter

    try:
        from cuml.cluster import KMeans as cluster_factory
    except Exception:
        from sklearn.cluster import KMeans as cluster_factory

    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    wcss_values = {}
    silhouette_scores = {}
    davies_bouldin_indices = {}
    calinski_harabasz_indices = {}
    suggested_k_elbow = None
    suggested_k_silhouette = None
    suggested_k_davies_bouldin = None
    suggested_k_calinski_harabasz = None
    final_suggestion_k = None
    kmeans_lib = cluster_factory
    normalized_k_range = tuple(k_range)
    k_report = create_activity_reporter(len(normalized_k_range) or 1)
    is_cupy_available = getattr(np, "__name__", "").lower() == "cupy"
    if is_cupy_available and (kmeans_lib is not None):
        print(
            "GPU acceleration with CuPy (cuML) is available and will be used."
        )
    else:
        print(
            "Warning: CuPy (cuML) is unavailable, falling back to CPU with scikit-learn KMeans."
        )
    X_array = np.asarray(X)
    if len(normalized_k_range) < 2:
        return {
            "wcss": wcss_values,
            "silhouette_scores": silhouette_scores,
            "davies_bouldin_indices": davies_bouldin_indices,
            "calinski_harabasz_indices": calinski_harabasz_indices,
            "suggested_k_elbow": suggested_k_elbow,
            "suggested_k_silhouette": suggested_k_silhouette,
            "suggested_k_davies_bouldin": suggested_k_davies_bouldin,
            "suggested_k_calinski_harabasz": suggested_k_calinski_harabasz,
            "final_suggestion": final_suggestion_k,
            "notes": "K-range too small to provide meaningful suggestions. Try a range with at least 2 different k values.",
        }
    for index, k in enumerate(normalized_k_range, start=1):
        k_report(
            index,
            "Score cluster count",
            detail=f"Evaluating k={k} ({index}/{len(normalized_k_range)}).",
        )
        if k <= 1:
            wcss_values[k] = 0
            silhouette_scores[k] = np.nan
            davies_bouldin_indices[k] = np.nan
            calinski_harabasz_indices[k] = np.nan
            continue
        kmeans = kmeans_lib(
            n_clusters=int(k), random_state=random_state, init="k-means++"
        )
        labels = kmeans.fit_predict(X_array)
        numpy_labels = np.asnumpy(labels) if is_cupy_available else labels
        numpy_X = np.asnumpy(X_array) if is_cupy_available else X_array
        wcss_values[k] = kmeans.inertia_
        silhouette_scores[k] = silhouette_score(numpy_X, numpy_labels)
        davies_bouldin_indices[k] = davies_bouldin_score(numpy_X, numpy_labels)
        calinski_harabasz_indices[k] = calinski_harabasz_score(
            numpy_X, numpy_labels
        )
    wcss_ratios = {}
    if len(normalized_k_range) > 2:
        for i in range(len(normalized_k_range) - 1):
            k1 = normalized_k_range[i]
            k2 = normalized_k_range[i + 1]
            if wcss_values[k1] > 0:
                ratio = wcss_values[k2] / wcss_values[k1]
                wcss_ratios[k2] = ratio
        if wcss_ratios:
            suggested_k_elbow = min(wcss_ratios, key=wcss_ratios.get)
    suggested_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
    suggested_k_davies_bouldin = min(
        davies_bouldin_indices, key=davies_bouldin_indices.get
    )
    suggested_k_calinski_harabasz = max(
        calinski_harabasz_indices, key=calinski_harabasz_indices.get
    )
    if suggested_k_elbow is not None:
        final_suggestion_k = suggested_k_elbow
    elif (
        suggested_k_silhouette is not None
        and silhouette_scores[suggested_k_silhouette] > 0.5
    ):
        final_suggestion_k = suggested_k_silhouette
    elif suggested_k_calinski_harabasz is not None:
        final_suggestion_k = suggested_k_calinski_harabasz
    else:
        final_suggestion_k = None
    return {
        "wcss": wcss_values,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_indices": davies_bouldin_indices,
        "calinski_harabasz_indices": calinski_harabasz_indices,
        "suggested_k_elbow": suggested_k_elbow,
        "suggested_k_silhouette": suggested_k_silhouette,
        "suggested_k_davies_bouldin": suggested_k_davies_bouldin,
        "suggested_k_calinski_harabasz": suggested_k_calinski_harabasz,
        "final_suggestion": final_suggestion_k,
        "random_state": random_state,
        "notes": "Suggestions are based on heuristics. Visualize metrics and use domain knowledge for final k selection. GPU acceleration is automatically used if available.",
    }


def get_model_instructions(task: str, model_type: str) -> str:
    import torch
    import torch.nn as nn

    from definers.constants import MODELS
    from definers.system import log

    del model_type
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        sklearn_available = True
    except ImportError:
        sklearn_available = False
    try:
        import onnxruntime

        onnx_available = True
    except ImportError:
        onnx_available = False
    profile = {
        "framework": "Unknown",
        "modalities": set(),
        "architecture": {"type": "Unknown", "details": []},
        "inputs": [],
        "outputs": [],
        "example_code": "",
        "notes": [],
    }

    def _analyze_architecture_pytorch(model_obj):
        if not isinstance(model_obj, nn.Module):
            return
        layer_counts = Counter(
            layer.__class__.__name__ for layer in model_obj.modules()
        )
        if (
            layer_counts["TransformerEncoderLayer"] > 0
            or layer_counts["MultiheadAttention"] > 0
        ):
            profile["architecture"]["type"] = "Transformer-based"
            if layer_counts["Conv2d"] > 2:
                profile["architecture"]["details"].append(
                    f"{layer_counts['TransformerEncoderLayer']} Transformer Blocks indicate a Vision Transformer (ViT) or hybrid architecture."
                )
            else:
                profile["architecture"]["details"].append(
                    f"{layer_counts['MultiheadAttention']} Attention Layers and {layer_counts['Embedding']} Embedding Layers form the core of this NLP/sequence model."
                )
        elif layer_counts["Conv2d"] > 4:
            profile["architecture"]["type"] = (
                "Convolutional Neural Network (CNN)"
            )
            profile["architecture"]["details"].extend(
                [
                    f"{layer_counts['Conv2d']} Conv2d layers",
                    f"{layer_counts['MaxPool2d']} Max-Pooling layers",
                    f"{layer_counts['Linear']} Fully-Connected layers",
                ]
            )
        elif layer_counts["Linear"] > 0:
            profile["architecture"]["type"] = "Multi-Layer Perceptron (MLP)"
            profile["architecture"]["details"].append(
                f"{layer_counts['Linear']} Linear layers"
            )

    def _probe_model_pytorch(model_obj):
        if not isinstance(model_obj, nn.Module):
            return
        try:
            sig = inspect.signature(model_obj.forward)
            dummy_inputs_kwargs = {}
            for param in sig.parameters.values():
                arg_name = param.name
                if arg_name in ["self", "args", "kwargs"]:
                    continue
                input_spec = next(
                    (
                        item
                        for item in profile["inputs"]
                        if item["name"] == arg_name
                    ),
                    None,
                )
                if not input_spec:
                    continue
                shape = tuple(
                    d if isinstance(d, int) else 2 for d in input_spec["shape"]
                )
                dtype_str = input_spec["dtype"]
                if "float" in dtype_str:
                    dummy_inputs_kwargs[arg_name] = torch.randn(
                        shape, dtype=getattr(torch, dtype_str)
                    )
                elif "long" in dtype_str or "int" in dtype_str:
                    vocab_size = next(
                        (
                            layer.num_embeddings
                            for layer in model_obj.modules()
                            if isinstance(layer, nn.Embedding)
                        ),
                        2000,
                    )
                    dummy_inputs_kwargs[arg_name] = torch.randint(
                        0, vocab_size, shape, dtype=torch.long
                    )
            if not dummy_inputs_kwargs:
                profile["notes"].append(
                    "Dynamic probe skipped: could not determine input arguments for `forward` method."
                )
                return
            model_obj.eval()
            with torch.no_grad():
                output = model_obj(**dummy_inputs_kwargs)
            output_tensors = (
                [output]
                if isinstance(output, torch.Tensor)
                else output
                if isinstance(output, (list, tuple))
                else []
            )
            for index, out_tensor in enumerate(output_tensors):
                if isinstance(out_tensor, torch.Tensor):
                    profile["outputs"].append(
                        {
                            "name": f"output_{index}",
                            "shape": tuple(out_tensor.shape),
                            "dtype": str(out_tensor.dtype).replace(
                                "torch.", ""
                            ),
                        }
                    )
            profile["notes"].append(
                "Dynamic probe SUCCESS: Input/Output specifications confirmed."
            )
        except Exception as error:
            profile["notes"].append(
                "Dynamic probe FAILED: Model `forward` pass raised an error, which may indicate complex input requirements not automatically detectable. "
                f"Error: {error}"
            )

    def _generate_report():
        modalities_str = (
            ", ".join(sorted([m.capitalize() for m in profile["modalities"]]))
            if profile["modalities"]
            else "Undetermined"
        )
        report = f"## Model Deep Dive Analysis: `{task}`\n\n"
        report += f"**Framework**: `{profile['framework']}`\n"
        report += f"**Detected Modality**: `{modalities_str}`\n"
        report += (
            f"**Detected Architecture**: `{profile['architecture']['type']}`\n"
        )
        if profile["architecture"]["details"]:
            details = "\n".join(
                [f"- {detail}" for detail in profile["architecture"]["details"]]
            )
            report += f"**Architectural Details**:\n{details}\n"
        report += "\n---\n### Input & Output Specification\n"
        if not profile["inputs"]:
            report += "**Inputs**: Could not be determined automatically.\n"
        for index, inp in enumerate(profile["inputs"]):
            report += (
                f"- **INPUT `{index}` (`{inp.get('name', 'N/A')}`)**: "
                f"Shape=`{inp['shape']}`, DType=`{inp['dtype']}`\n"
            )
        if not profile["outputs"]:
            report += "**Outputs**: Not confirmed. Dynamic probe did not run or failed.\n"
        for index, out in enumerate(profile["outputs"]):
            report += (
                f"- **OUTPUT `{index}` (`{out.get('name', 'N/A')}`)**: "
                f"Shape=`{out['shape']}`, DType=`{out['dtype']}` (Confirmed by probe)\n"
            )
        report += "\n---\n### Preprocessing & Usage Guide\n"
        example_imports = ""
        prep_steps = ""
        example_body = ""
        if profile["framework"] == "PyTorch":
            example_imports = "import torch\n"
            example_body = (
                f"model = YourModelClass() # Instantiate your defined model architecture\n"
                f"model.load_state_dict(torch.load('path/to/{task}.pt'))\n"
                "model.eval()\n\n"
                "dummy_inputs = {}\n"
            )
            for inp in profile["inputs"]:
                shape = inp["shape"]
                dtype = inp["dtype"]
                name = inp["name"]
                if "image" in name or "pixel" in name:
                    channels, height, width = (
                        shape[1],
                        shape[2],
                        shape[3],
                    )
                    prep_steps += (
                        f"**For Input `{name}` (Image)**:\n"
                        "1. Load image (e.g., with Pillow).\n"
                        f"2. Resize to `{height}x{width}`.\n"
                        "3. Convert to a tensor and normalize (e.g., ImageNet stats).\n"
                        f"4. Ensure shape is `(1, {channels}, {height}, {width})`.\n"
                    )
                    example_imports += (
                        "import numpy as np\nfrom PIL import Image\n"
                    )
                    example_body += "image = Image.open('path/to/image.jpg').convert('RGB')\n"
                    example_body += (
                        f"image = image.resize(({width}, {height}))\n"
                    )
                    example_body += "image_array = np.asarray(image, dtype=np.float32) / 255.0\n"
                    example_body += "image_array = (image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)\n"
                    example_body += f"dummy_inputs['{name}'] = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)\n"
                elif "text" in name or "ids" in name:
                    prep_steps += (
                        f"**For Input `{name}` (Text)**:\n"
                        "1. Use the specific tokenizer the model was trained with.\n"
                        "2. Convert text to token IDs.\n"
                        f"3. Format as a `{dtype}` tensor with shape `{shape}`.\n"
                    )
                    example_body += f"# Use the model's specific tokenizer\ndummy_inputs['{name}'] = torch.randint(0, 1000, {shape}, dtype=torch.long)\n"
            example_body += (
                "\nwith torch.no_grad():\n"
                "    output = model(**dummy_inputs)\n"
                "    print(f'Output: {output}')\n"
            )
            profile["example_code"] = (
                f"```python\n{example_imports}\n# 1. Define or import your model class (YourModelClass)\n\n# 2. Load model and prepare inputs\n{example_body}```"
            )
        elif profile["framework"] == "scikit-learn":
            prep_steps += (
                "1. Ensure your input data is in the correct format (NumPy array, Pandas DataFrame, or raw text for pipelines).\n"
                "2. Apply the exact same feature engineering and preprocessing steps used during training.\n"
            )
            example_body = (
                f"import joblib\nmodel = joblib.load('path/to/{task}.pkl')\n"
            )
            if "TfidfVectorizer" in profile["architecture"]["details"]:
                example_body += "input_data = ['your first sentence', 'your second sentence']\n"
            else:
                n_features = profile["inputs"][0]["shape"][1]
                example_body += (
                    "import numpy as np\n"
                    f"# Input must be a 2D array with shape (n_samples, {n_features})\n"
                    f"input_data = np.random.rand(2, {n_features})\n"
                )
            example_body += (
                "predictions = model.predict(input_data)\nprint(predictions)"
            )
            profile["example_code"] = f"```python\n{example_body}\n```"
        report += (
            prep_steps
            + "\n#### Example Usage Snippet\n"
            + profile["example_code"]
        )
        if profile["notes"]:
            report += "\n---\n### Analyst Notes\n" + "\n".join(
                [f"- {note}" for note in profile["notes"]]
            )
        return report

    model_object = MODELS.get(task)
    if model_object is None:
        log(
            f"Analysis Failed for '{task}'",
            f"Model file `{task}` could not be found or loaded.",
            status=False,
        )
        return
    if isinstance(model_object, nn.Module) or isinstance(
        model_object, (dict, OrderedDict)
    ):
        profile["framework"] = "PyTorch"
        if isinstance(model_object, (dict, OrderedDict)):
            profile["architecture"]["type"] = "Raw State Dictionary"
            profile["notes"].append(
                "Analysis is based on a state_dict, not a full model. Instantiate the model class before loading these weights."
            )
            first_key, first_tensor = next(iter(model_object.items()))
            del first_key
            in_features = (
                first_tensor.shape[1] if len(first_tensor.shape) == 2 else "N/A"
            )
            profile["inputs"].append(
                {
                    "name": "input_0",
                    "shape": f"(batch_size, {in_features})",
                    "dtype": str(first_tensor.dtype),
                }
            )
            profile["modalities"].add(
                "Tabular" if in_features != "N/A" else "Unknown"
            )
        else:
            sig = inspect.signature(model_object.forward)
            for param in sig.parameters.values():
                if param.name in ["self", "args", "kwargs"]:
                    continue
                name = param.name
                if "image" in name or "pixel" in name:
                    profile["modalities"].add("Image")
                    profile["inputs"].append(
                        {
                            "name": name,
                            "shape": (1, 3, 32, 32),
                            "dtype": "float32",
                        }
                    )
                elif "text" in name or "ids" in name:
                    profile["modalities"].add("Text")
                    profile["inputs"].append(
                        {"name": name, "shape": (1, 16), "dtype": "long"}
                    )
                else:
                    profile["modalities"].add("Tabular")
                    profile["inputs"].append(
                        {
                            "name": name,
                            "shape": (1, 64),
                            "dtype": "float32",
                        }
                    )
            _analyze_architecture_pytorch(model_object)
            _probe_model_pytorch(model_object)
    elif sklearn_available and hasattr(model_object, "predict"):
        profile["framework"] = "scikit-learn"
        if isinstance(model_object, Pipeline):
            profile["architecture"]["type"] = "Scikit-learn Pipeline"
            steps = [
                f"{name} ({step.__class__.__name__})"
                for name, step in model_object.steps
            ]
            profile["architecture"]["details"] = steps
            if any("TfidfVectorizer" in step for step in steps):
                profile["modalities"].add("Text")
                profile["inputs"].append(
                    {
                        "name": "raw_text",
                        "shape": "(n_samples,)",
                        "dtype": "string",
                    }
                )
        else:
            profile["architecture"]["type"] = (
                f"Standard Model ({model_object.__class__.__name__})"
            )
            profile["modalities"].add("Tabular")
            n_features = getattr(model_object, "n_features_in_", "N/A")
            profile["inputs"].append(
                {
                    "name": "X",
                    "shape": f"(n_samples, {n_features})",
                    "dtype": "float",
                }
            )
    elif onnx_available and isinstance(
        model_object, onnxruntime.InferenceSession
    ):
        profile["framework"] = "ONNX"
        profile["architecture"]["type"] = "ONNX Graph"
        for inp in model_object.get_inputs():
            profile["inputs"].append(
                {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
            )
            if len(inp.shape) == 4 and inp.shape[1] in [1, 3]:
                profile["modalities"].add("Image")
        for out in model_object.get_outputs():
            profile["outputs"].append(
                {"name": out.name, "shape": out.shape, "dtype": out.type}
            )
    final_report = _generate_report()
    log(f"Deep Dive Analysis for '{task}'", final_report)


def compile_model(model_or_pipeline):
    import inspect
    import types

    import torch

    try:
        from diffusers import DiffusionPipeline
        from diffusers.models.modeling_utils import ModelMixin
        from transformers import PreTrainedModel
    except ImportError:
        warnings.warn(
            "Please install `diffusers` and `transformers` for full functionality."
        )
        return model_or_pipeline
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile() is not available. Please use PyTorch 2.0 or newer."
        )
        return model_or_pipeline
    compile_kwargs = {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
    }

    def patch_forward(obj):
        if not hasattr(obj, "forward"):
            return obj
        orig_forward = obj.forward
        src = inspect.getsource(orig_forward)
        if ".to(sample.device)" in src:
            patched_src = src.replace(
                ".to(sample.device)", ".to(sample.device.type)"
            )
            globals_dict = orig_forward.__globals__.copy()
            exec(patched_src, globals_dict)
            new_forward = globals_dict[orig_forward.__name__]
            obj.forward = types.MethodType(new_forward, obj)
            print(
                f"Patched {type(obj).__name__}.forward to avoid .to(sample.device) bug for torch.compile."
            )
        return obj

    if isinstance(model_or_pipeline, DiffusionPipeline):
        print(
            "Detected a Diffusers pipeline. Dynamically compiling submodels..."
        )
        for attr_name, attr_value in model_or_pipeline.__dict__.items():
            if isinstance(attr_value, ModelMixin):
                try:
                    attr_value = patch_forward(attr_value)
                    print(f"   -> Compiling {attr_name}...")
                    if attr_name == "vae":
                        attr_value.decode = torch.compile(
                            attr_value.decode, **compile_kwargs
                        )
                    else:
                        setattr(
                            model_or_pipeline,
                            attr_name,
                            torch.compile(attr_value, **compile_kwargs),
                        )
                except Exception as error:
                    warnings.warn(
                        f"Could not compile submodel '{attr_name}'. Reason: {error}"
                    )
        return model_or_pipeline
    if isinstance(model_or_pipeline, PreTrainedModel):
        print("Detected a Transformers model. Compiling the model...")
        try:
            return torch.compile(model_or_pipeline, **compile_kwargs)
        except Exception as error:
            warnings.warn(f"Could not compile the model. Reason: {error}")
            return model_or_pipeline
    warnings.warn(
        "Object is not a recognized Diffusers pipeline or Transformers model. No action taken."
    )
    return model_or_pipeline


__all__ = [
    "compile_model",
    "get_model_instructions",
    "kmeans_k_suggestions",
]
