from __future__ import annotations


def normalize_selected_rows(selected_rows):
    import gradio as gr

    from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
    from definers.ml import simple_text

    if selected_rows is None:
        return None
    if len(selected_rows) > MAX_INPUT_LENGTH:
        raise gr.Error(
            f"Selected rows input too long ({len(selected_rows)} > {MAX_INPUT_LENGTH})"
        )
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in selected_rows:
        raise gr.Error("Selected rows contains too many consecutive spaces")
    return simple_text(selected_rows)


def build_training_plan_markdown(
    features,
    labels,
    model_path,
    remote_src,
    dataset_label_columns,
    revision,
    url_type,
    drop_list,
    selected_rows,
):
    from definers.application_ml.trainer_plan import (
        render_training_plan_markdown,
    )
    from definers.ml import AutoTrainer

    trainer = AutoTrainer(revision=revision, source_type=url_type or "parquet")
    normalized_selected_rows = normalize_selected_rows(selected_rows)
    plan = trainer.training_plan(
        data=remote_src or features,
        target=labels,
        resume_from=model_path,
        label_columns=dataset_label_columns,
        drop=drop_list,
        select=normalized_selected_rows,
    )
    return render_training_plan_markdown(plan)


def handle_training(
    features,
    labels,
    model_path,
    remote_src,
    dataset_label_columns,
    revision,
    url_type,
    drop_list,
    selected_rows,
):
    from definers.ml import AutoTrainer

    trainer = AutoTrainer(revision=revision, source_type=url_type or "parquet")
    normalized_selected_rows = normalize_selected_rows(selected_rows)
    plan_markdown = build_training_plan_markdown(
        features,
        labels,
        model_path,
        remote_src,
        dataset_label_columns,
        revision,
        url_type,
        drop_list,
        normalized_selected_rows,
    )
    model_output = trainer.train(
        data=remote_src or features,
        target=labels,
        save_as=model_path,
        revision=revision,
        source_type=url_type,
        label_columns=dataset_label_columns,
        drop=drop_list,
        select=normalized_selected_rows,
        resume_from=model_path,
    )
    return model_output, plan_markdown


def handle_prediction(model_predict, prediction_data):
    from definers.ml import AutoTrainer

    trainer = AutoTrainer(model_path=model_predict)
    return trainer.predict(prediction_data)
