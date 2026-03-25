class ApplicationDataFacade:
    @staticmethod
    def contracts_module():
        import definers.application_data.contracts as contracts_module

        return contracts_module

    @classmethod
    def dataset_loader_port(cls):
        return cls.contracts_module().DatasetLoaderPort

    @classmethod
    def tensor_adapter_port(cls):
        return cls.contracts_module().TensorAdapterPort

    @classmethod
    def vectorization_port(cls):
        return cls.contracts_module().VectorizationPort


DatasetLoaderPort = ApplicationDataFacade.dataset_loader_port()
TensorAdapterPort = ApplicationDataFacade.tensor_adapter_port()
VectorizationPort = ApplicationDataFacade.vectorization_port()

__all__ = ["DatasetLoaderPort", "TensorAdapterPort", "VectorizationPort"]
