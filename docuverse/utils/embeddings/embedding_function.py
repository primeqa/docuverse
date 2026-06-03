import os

from typing import Union, List
from docuverse.utils import get_param, get_config_dir, convert_to_type

class EmbeddingFunction:
    def __init__(self, model_or_directory_name: str, batch_size: int, **kwargs):
        self.model = None
        self.batch_size = batch_size
        self.matryoshka_dim = int(kwargs.get("matryoshka_dim", 0) or 0)
        self.warmup_batches = int(kwargs.get("warmup_batches", 0) or 0)
        self._warmup_done = False
        import torch
        torch.set_float32_matmul_precision('high')
        self.torch_dtype = convert_to_type(kwargs.get("model_torch_dtype", None))
        self.torch_compile = bool(kwargs.get("torch_compile", False))
        if self.torch_compile:
            import logging
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)


    @staticmethod
    def pull_from_dmf(name):
        try:
            from lakehouse.wrappers import LakehouseIceberg
            from lakehouse.assets import Model
        except ImportError as e:
            print(f"You need to install the DMF library:\n"
                  f"\tpip install git+ssh://git@github.ibm.com/arc/dmf-library.git")
            raise e
        lh = LakehouseIceberg(config="yaml", conf_location=get_config_dir("config/lh-conf.yaml"))
        model = Model(lh=lh)
        if get_param(os.environ, 'DMF_MODEL_CACHE', None) is None:
            os.environ['DMF_MODEL_CACHE'] = os.path.join(os.environ['HOME'], ".local/share/dmf")
        name = model.pull(model=name, namespace="retrieval_slate", table="model_shared")
        return name


    def __call__(self, texts:Union[List[str], str], **kwargs) -> \
        Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts, **kwargs)

    def __del__(self):
        pass

    def create_model(self, model_or_directory_name:str=None, device: str='cpu', **kwargs):
        pass

    def tokenizer(self):
        return self.model.tokenizer

    def start_pool(self):
        pass

    def stop_pool(self):
        pass

    def encode(self, texts:Union[List[str], str], _batch_size:int=-1,
               show_progress_bar=False, tqdm_instance=None, prompt_name=None, tm=None, **kwargs):
        return []

    def encode_query(self, texts: Union[List[str], str], prompt_name:str|None=None, tm=None, **kwargs):
        return []

    # def encode_query(self, texts:Union[List[str], str], _batch_size:int=-1,
    #                  show_progress_bar=False, tqdm_instance=None, **kwargs):
    #     return []

    @staticmethod
    def print_gpu_stats(label=""):
        import torch
        if not torch.cuda.is_available():
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            has_nvml = True
        except (ImportError, pynvml.NVMLError):
            has_nvml = False

        n_gpus = torch.cuda.device_count()
        rows = []
        for i in range(n_gpus):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            row = {
                "GPU": f"{i}: {torch.cuda.get_device_name(i)}",
                "Alloc (MB)": f"{alloc:.0f}",
                "Rsrvd (MB)": f"{reserved:.0f}",
                "Total (MB)": f"{total:.0f}",
            }
            if has_nvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    max_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                    row["Clock (MHz)"] = f"{clock}/{max_clock}"
                    row["Power (W)"] = f"{power:.0f}/{power_limit:.0f}"
                except pynvml.NVMLError:
                    pass
            rows.append(row)

        if has_nvml:
            pynvml.nvmlShutdown()

        if not rows:
            return

        # Compute column widths and print table
        headers = list(rows[0].keys())
        widths = {h: max(len(h), *(len(r.get(h, "")) for r in rows)) for h in headers}
        sep = "+-" + "-+-".join("-" * widths[h] for h in headers) + "-+"
        hdr = "| " + " | ".join(h.rjust(widths[h]) if h != "GPU" else h.ljust(widths[h]) for h in headers) + " |"
        title = f"[{label}]" if label else "GPU Stats"
        print(title)
        print(sep)
        print(hdr)
        print(sep)
        for row in rows:
            cells = []
            for h in headers:
                val = row.get(h, "")
                cells.append(val.ljust(widths[h]) if h == "GPU" else val.rjust(widths[h]))
            print("| " + " | ".join(cells) + " |")
        print(sep)

    @property
    def vocab_size(self):
        return self.model.tokenizer.vocab_size

    @property
    def device(self):
        return self.model.model.device