from __future__ import annotations

from definers.constants import language_codes


def lang_code_to_name(code: str):
    if code in language_codes:
        return language_codes[code]
    lower_code = code.lower()
    if lower_code in language_codes:
        return language_codes[lower_code]
    raise KeyError(code)


def get_cluster_content(model, cluster_index):
    if not hasattr(model, "labels_"):
        raise ValueError("Model must be a trained KMeans model.")
    cluster_labels = model.labels_
    model_rows = getattr(model, "x_all", getattr(model, "X_all", None))
    cluster_contents = {}
    for index, label in enumerate(cluster_labels):
        if label not in cluster_contents:
            cluster_contents[label] = []
        if model_rows is not None:
            cluster_contents[label].append(model_rows[index])
    return cluster_contents.get(cluster_index)


def is_clusters_model(model) -> bool:
    if model is None or isinstance(model, (str, bytes)):
        return False
    try:
        model_vars = vars(model)
    except Exception:
        return False
    return (
        "cluster_centers_" in model_vars
        and model_vars["cluster_centers_"] is not None
    )
