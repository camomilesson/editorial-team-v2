"""Checkpoint construction for the disconnected graph foundation."""

import pickle
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class _ProcessLocalSerializer:
    """Round-trip trusted objects held only by this process's memory saver."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_name, payload = data
        if type_name != "pickle":
            raise ValueError("Unsupported process-local checkpoint payload")
        return pickle.loads(payload)


def create_in_memory_checkpointer() -> InMemorySaver:
    """Return a fresh process-local checkpointer for one composed application.

    The foundation is not connected to application composition yet. A later
    milestone will own one returned instance for the lifetime of an application.
    """

    return InMemorySaver(serde=_ProcessLocalSerializer())
