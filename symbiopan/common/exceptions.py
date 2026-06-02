class SymbioPanError(Exception):
    pass


class CheckpointMismatchError(SymbioPanError):
    pass


class DataLeakageError(SymbioPanError):
    pass
