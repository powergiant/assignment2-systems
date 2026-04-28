import timeit
from torch.cuda import synchronize

_is_init: bool = False
_record: list[tuple[str, int, int]] = []

def init_profiler():
    global _is_init, _record

def is_init() -> bool:
    return _is_init

def get_result() -> list:
    if _is_init:
        return _record
    else:
        raise "Must initialize before using profiler!"

def find(name: str) -> tuple[int, int]:
    for name_in_record, t_start, t_end in _record:
        if name == name_in_record:
            return t_start, t_end
    raise f"There is no name {name} in the record!"

class TorchStepProfiler:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        global _record
        _record.append((self.name if self.name else "", timeit.default_timer(), None))

    def __exit__(self, *args):
        global _record
        synchronize()
        _record[-1] = (_record[-1][0], timeit.default_timer())