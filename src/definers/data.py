class DataFacade:
    @staticmethod
    def arrays_module():
        import definers.application_data.arrays as arrays_module

        return arrays_module

    @staticmethod
    def exports_module():
        import definers.application_data.exports as exports_module

        return exports_module

    @staticmethod
    def loaders_module():
        import definers.application_data.loaders as loaders_module

        return loaders_module

    @staticmethod
    def preparation_module():
        import definers.application_data.preparation as preparation_module

        return preparation_module

    @staticmethod
    def runtime_patches_module():
        import definers.application_data.runtime_patches as runtime_patches_module

        return runtime_patches_module

    @staticmethod
    def tokenization_module():
        import definers.application_data.tokenization as tokenization_module

        return tokenization_module

    @staticmethod
    def vectorizers_module():
        import definers.application_data.vectorizers as vectorizers_module

        return vectorizers_module

    @staticmethod
    def logger_instance():
        from definers.logger import init_logger

        return init_logger()

    @classmethod
    def numpy_modules(cls):
        return cls.runtime_patches_module().init_cupy_numpy()

    @classmethod
    def training_data(cls):
        return cls.preparation_module().TrainingData

    @classmethod
    def convert_tensor_dtype(cls):
        return cls.arrays_module().convert_tensor_dtype

    @classmethod
    def cupy_to_numpy(cls):
        return cls.arrays_module().cupy_to_numpy

    @classmethod
    def dtype(cls):
        return cls.arrays_module().dtype

    @classmethod
    def get_max_shapes(cls):
        return cls.arrays_module().get_max_shapes

    @classmethod
    def guess_numpy_sample_rate(cls):
        return cls.arrays_module().guess_numpy_sample_rate

    @classmethod
    def guess_numpy_type(cls):
        return cls.arrays_module().guess_numpy_type

    @classmethod
    def infer_data_type(cls):
        return cls.arrays_module().infer_data_type

    @classmethod
    def numpy_to_cupy(cls):
        return cls.arrays_module().numpy_to_cupy

    @classmethod
    def numpy_to_list(cls):
        return cls.arrays_module().numpy_to_list

    @classmethod
    def numpy_to_str(cls):
        return cls.arrays_module().numpy_to_str

    @classmethod
    def one_dim_numpy(cls):
        return cls.arrays_module().one_dim_numpy

    @classmethod
    def pad_nested(cls):
        return cls.arrays_module().pad_nested

    @classmethod
    def pad_or_reshape(cls):
        return cls.arrays_module().pad_or_reshape

    @classmethod
    def reshape_numpy(cls):
        return cls.arrays_module().reshape_numpy

    @classmethod
    def str_to_numpy(cls):
        return cls.arrays_module().str_to_numpy

    @classmethod
    def tensor_length(cls):
        return cls.arrays_module().tensor_length

    @classmethod
    def three_dim_numpy(cls):
        return cls.arrays_module().three_dim_numpy

    @classmethod
    def two_dim_numpy(cls):
        return cls.arrays_module().two_dim_numpy

    @classmethod
    def check_onnx(cls):
        return cls.exports_module().check_onnx

    @classmethod
    def get_prediction_file_extension(cls):
        return cls.exports_module().get_prediction_file_extension

    @classmethod
    def is_gpu(cls):
        return cls.exports_module().is_gpu

    @classmethod
    def pytorch_to_onnx(cls):
        return cls.exports_module().pytorch_to_onnx

    @classmethod
    def read_as_numpy(cls):
        return cls.exports_module().read_as_numpy

    @classmethod
    def drop_columns(cls):
        return cls.loaders_module().drop_columns

    @classmethod
    def fetch_dataset(cls):
        return cls.loaders_module().fetch_dataset

    @classmethod
    def files_to_dataset(cls):
        return cls.loaders_module().files_to_dataset

    @classmethod
    def load_as_numpy(cls):
        return cls.loaders_module().load_as_numpy

    @classmethod
    def load_source(cls):
        return cls.loaders_module().load_source

    @classmethod
    def select_columns(cls):
        return cls.loaders_module().select_columns

    @classmethod
    def select_rows(cls):
        return cls.loaders_module().select_rows

    @classmethod
    def split_columns(cls):
        return cls.loaders_module().split_columns

    @classmethod
    def make_loader(cls):
        return cls.preparation_module().make_loader

    @classmethod
    def merge_columns(cls):
        return cls.preparation_module().merge_columns

    @classmethod
    def order_dataset(cls):
        return cls.preparation_module().order_dataset

    @classmethod
    def pad_sequences(cls):
        return cls.preparation_module().pad_sequences

    @classmethod
    def prepare_data(cls):
        return cls.preparation_module().prepare_data

    @classmethod
    def process_rows(cls):
        return cls.preparation_module().process_rows

    @classmethod
    def split_dataset(cls):
        return cls.preparation_module().split_dataset

    @classmethod
    def to_loader(cls):
        return cls.preparation_module().to_loader

    @classmethod
    def init_cupy_numpy(cls):
        return cls.runtime_patches_module().init_cupy_numpy

    @classmethod
    def init_tokenizer(cls):
        return cls.tokenization_module().init_tokenizer

    @classmethod
    def tokenize_and_pad(cls):
        return cls.tokenization_module().tokenize_and_pad

    @classmethod
    def tokenize_or_vectorize(cls):
        return cls.tokenization_module().tokenize_or_vectorize

    @classmethod
    def create_vectorizer(cls):
        return cls.vectorizers_module().create_vectorizer

    @classmethod
    def unvectorize(cls):
        return cls.vectorizers_module().unvectorize

    @classmethod
    def vectorize(cls):
        return cls.vectorizers_module().vectorize


logger = DataFacade.logger_instance()
np, _np = DataFacade.numpy_modules()
TrainingData = DataFacade.training_data()
check_onnx = DataFacade.check_onnx()
convert_tensor_dtype = DataFacade.convert_tensor_dtype()
create_vectorizer = DataFacade.create_vectorizer()
cupy_to_numpy = DataFacade.cupy_to_numpy()
dtype = DataFacade.dtype()
drop_columns = DataFacade.drop_columns()
fetch_dataset = DataFacade.fetch_dataset()
files_to_dataset = DataFacade.files_to_dataset()
get_max_shapes = DataFacade.get_max_shapes()
get_prediction_file_extension = DataFacade.get_prediction_file_extension()
guess_numpy_sample_rate = DataFacade.guess_numpy_sample_rate()
guess_numpy_type = DataFacade.guess_numpy_type()
infer_data_type = DataFacade.infer_data_type()
init_cupy_numpy = DataFacade.init_cupy_numpy()
init_tokenizer = DataFacade.init_tokenizer()
is_gpu = DataFacade.is_gpu()
load_as_numpy = DataFacade.load_as_numpy()
load_source = DataFacade.load_source()
make_loader = DataFacade.make_loader()
merge_columns = DataFacade.merge_columns()
numpy_to_cupy = DataFacade.numpy_to_cupy()
numpy_to_list = DataFacade.numpy_to_list()
numpy_to_str = DataFacade.numpy_to_str()
one_dim_numpy = DataFacade.one_dim_numpy()
order_dataset = DataFacade.order_dataset()
pad_nested = DataFacade.pad_nested()
pad_or_reshape = DataFacade.pad_or_reshape()
pad_sequences = DataFacade.pad_sequences()
prepare_data = DataFacade.prepare_data()
process_rows = DataFacade.process_rows()
pytorch_to_onnx = DataFacade.pytorch_to_onnx()
read_as_numpy = DataFacade.read_as_numpy()
reshape_numpy = DataFacade.reshape_numpy()
select_columns = DataFacade.select_columns()
select_rows = DataFacade.select_rows()
split_columns = DataFacade.split_columns()
split_dataset = DataFacade.split_dataset()
str_to_numpy = DataFacade.str_to_numpy()
tensor_length = DataFacade.tensor_length()
three_dim_numpy = DataFacade.three_dim_numpy()
to_loader = DataFacade.to_loader()
tokenize_and_pad = DataFacade.tokenize_and_pad()
tokenize_or_vectorize = DataFacade.tokenize_or_vectorize()
two_dim_numpy = DataFacade.two_dim_numpy()
unvectorize = DataFacade.unvectorize()
vectorize = DataFacade.vectorize()

__all__ = [
    "TrainingData",
    "check_onnx",
    "convert_tensor_dtype",
    "create_vectorizer",
    "cupy_to_numpy",
    "dtype",
    "drop_columns",
    "fetch_dataset",
    "files_to_dataset",
    "get_max_shapes",
    "get_prediction_file_extension",
    "guess_numpy_sample_rate",
    "guess_numpy_type",
    "infer_data_type",
    "init_tokenizer",
    "is_gpu",
    "load_as_numpy",
    "load_source",
    "make_loader",
    "merge_columns",
    "numpy_to_cupy",
    "numpy_to_list",
    "numpy_to_str",
    "one_dim_numpy",
    "order_dataset",
    "pad_nested",
    "pad_or_reshape",
    "pad_sequences",
    "prepare_data",
    "process_rows",
    "pytorch_to_onnx",
    "read_as_numpy",
    "reshape_numpy",
    "select_columns",
    "select_rows",
    "split_columns",
    "split_dataset",
    "str_to_numpy",
    "tensor_length",
    "three_dim_numpy",
    "to_loader",
    "tokenize_and_pad",
    "tokenize_or_vectorize",
    "two_dim_numpy",
    "unvectorize",
    "vectorize",
]
