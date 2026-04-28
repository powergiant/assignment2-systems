import timeit
from torch.cuda import synchronize

_is_init = False
_record = []

def init_profiler():
    global _is_init, _record

def is_init() -> bool:
    return _is_init

def get_result() -> list:
    if _is_init:
        return _record
    else:
        raise "Must initialize before using profiler!"

class TorchStepProfiler:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        global _record
        _record.append((self.name if self.name else "", timeit.default_timer(), None))

    def __exit__(self, *args):
        global _record
        synchronize()
        _record[-1][1] = timeit.default_timer()