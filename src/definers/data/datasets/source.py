class DatasetSourceLoader:
    @staticmethod
    def load_remote_dataset(
        src: str,
        revision: str | None,
        sample_rows: int | None = None,
    ):
        from datasets import load_dataset

        split_name = "train"
        if sample_rows is not None and sample_rows > 0:
            split_name = f"train[:{int(sample_rows)}]"
        if revision:
            return load_dataset(src, revision=revision, split=split_name)
        return load_dataset(src, split=split_name)

    @staticmethod
    def load_remote_dataset_fallback(
        src: str,
        url_type: str,
        revision: str | None,
        sample_rows: int | None = None,
    ):
        from datasets import load_dataset

        split_name = "train"
        if sample_rows is not None and sample_rows > 0:
            split_name = f"train[:{int(sample_rows)}]"
        if revision:
            return load_dataset(
                url_type,
                data_files={"train": src},
                revision=revision,
                split=split_name,
            )
        return load_dataset(
            url_type,
            data_files={"train": src},
            split=split_name,
        )

    @staticmethod
    def load_dataset_attempt(load_dataset_call, source_name: str):
        try:
            return load_dataset_call(), True
        except FileNotFoundError:
            import logging

            logging.error(f"Dataset {source_name} not found.")
            return None, False
        except ConnectionError:
            import logging

            logging.error(
                f"Connection error while loading dataset {source_name}."
            )
            return None, False
        except Exception as error:
            import logging

            logging.error(f"Error loading dataset {source_name}: {error}")
            return None, True

    @staticmethod
    def fetch_dataset(
        src: str,
        url_type: str | None = None,
        revision: str | None = None,
        sample_rows: int | None = None,
    ):
        import definers.data.loaders as loaders_module

        dataset, allow_fallback = loaders_module._load_dataset_attempt(
            lambda: loaders_module._load_remote_dataset(
                src,
                revision,
                sample_rows=sample_rows,
            ),
            src,
        )
        if dataset is not None or url_type is None or not allow_fallback:
            return dataset
        fallback_source = f"{url_type} with data_files {src}"
        fallback_dataset, _ = loaders_module._load_dataset_attempt(
            lambda: loaders_module._load_remote_dataset_fallback(
                src,
                url_type,
                revision,
                sample_rows=sample_rows,
            ),
            fallback_source,
        )
        return fallback_dataset

    @staticmethod
    def load_source(
        remote_src: str | None = None,
        features=None,
        labels=None,
        url_type: str | None = None,
        revision: str | None = None,
    ):
        import definers.data.loaders as loaders_module

        runtime = loaders_module._runtime()
        if remote_src:
            fetch = getattr(
                runtime, "fetch_dataset", loaders_module.fetch_dataset
            )
            return fetch(remote_src, url_type, revision)
        if features:
            build_dataset = getattr(
                runtime, "files_to_dataset", loaders_module.files_to_dataset
            )
            return build_dataset(features, labels)
        return None
