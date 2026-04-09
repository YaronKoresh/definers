def has_parameter(values) -> bool:
    return values is not None and not (
        isinstance(values, list)
        and (
            len(values) == 0
            or (isinstance(values[0], str) and values[0].strip() == "")
        )
    )


def column_names(dataset) -> tuple[str, ...]:
    return tuple(getattr(dataset, "column_names", ()))


def selected_columns(dataset, requested_columns) -> list[str]:
    requested_names = set(requested_columns)
    return [
        column_name
        for column_name in column_names(dataset)
        if column_name in requested_names
    ]


def dataset_slice(
    dataset, start_index: int, end_index: int
) -> dict[str, object]:
    return {
        column_name: dataset[column_name][start_index:end_index]
        for column_name in column_names(dataset)
    }


def batched_column_values(data, label_names):
    import numpy as np

    features_batch = []
    labels_batch = []
    batch_size = 0
    for value in data.values():
        if isinstance(value, (list, np.ndarray)):
            batch_size = len(value)
            break
    if batch_size == 0:
        return [], []
    for index in range(batch_size):
        feature_row = {}
        label_row = {}
        for key, value in data.items():
            if key in label_names:
                label_row[key] = value[index]
            else:
                feature_row[key] = value[index]
        features_batch.append(feature_row)
        labels_batch.append(label_row)
    return features_batch, labels_batch


def path_extension(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) < 2:
        return None
    return parts[-1].strip().lower()


def drop_columns(dataset, drop_list):
    if not has_parameter(drop_list):
        return dataset
    columns_to_delete = selected_columns(dataset, drop_list)
    return dataset.remove_columns(columns_to_delete)


def select_columns(dataset, cols):
    if not has_parameter(cols):
        return dataset
    selected_names = set(cols)
    columns_to_drop = [
        column_name
        for column_name in column_names(dataset)
        if column_name not in selected_names
    ]
    return drop_columns(dataset, columns_to_drop)


def select_rows(dataset, start_index: int, end_index: int):
    from datasets import Dataset

    return Dataset.from_dict(dataset_slice(dataset, start_index, end_index))


def split_columns(data, labels, is_batch: bool = False):
    if not has_parameter(labels):
        features, label_values = data
        return features, label_values
    if is_batch:
        return batched_column_values(data, set(labels))
    features = drop_columns(data, labels)
    label_values = select_columns(data, labels)
    return features, label_values
