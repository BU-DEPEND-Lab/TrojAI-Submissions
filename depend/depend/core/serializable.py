"""Base interface for dependent module to expose."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr


from pandas import DataFrame
import pyarrow as pa
import jsonpickle


class BaseSerialized(TypedDict):
    """Base class for serialized objects."""

    dp: int
    id: List[str]


class SerializedConstructor(BaseSerialized):
    """Serialized constructor."""

    type: Literal["constructor"]
    kwargs: Dict[str, Any]
 


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented."""

    type: Literal["not_implemented"]


class Serializable(BaseModel, ABC):
    """Serializable base class."""

    @property
    def dp_serializable(self) -> bool:
        """
        Return whether or not the class is serializable.
        """
        return False

    @property
    def dp_namespace(self) -> List[str]:
        """
        Return the namespace of the dependent object.
        eg. ["dependent"]
        """
        return self.__class__.__module__.split(".")
 

    @property
    def dp_attributes(self) -> Dict:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {}

    class Config:
        extra = "ignore"

    _dp_kwargs = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dp_kwargs = kwargs

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        if not self.dp_serializable:
            return self.to_json_not_implemented()

        # Get latest values for kwargs if there is an attribute with same name
        dp_kwargs = {
            k: getattr(self, k, v)
            for k, v in self._dp_kwargs.items()
            if not (self.__exclude_fields__ or {}).get(k, False)  # type: ignore
        }

        # Merge the dp_secrets and dp_attributes from every class in the MRO
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            # Get a reference to self bound to each class in the MRO
            this = cast(Serializable, self if cls is None else super(cls, self))

            dp_kwargs.update(this.dp_attributes)

         
        return {
            "dp": 1,
            "type": "constructor",
            "id": [*self.dp_namespace, self.__class__.__name__],
            "kwargs": dp_kwargs
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return to_json_not_implemented(self)

def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: object to serialize

    Returns:
        SerializedNotImplemented
    """
    _id: List[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            _id = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        pass
    return {
        "dp": 1,
        "type": "not_implemented",
        "id": _id,
    }


    

def serialize_with_pyarrow(dataframe: DataFrame):
    batch = pa.record_batch(dataframe)
    write_options = pa.ipc.IpcWriteOptions(compression="zstd")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema,   options=write_options) as writer:
        writer.write_batch(batch)
    pybytes = sink.getvalue().to_pybytes()
    pybytes_str = jsonpickle.encode(pybytes, unpicklable=True, make_refs=False)
    return pybytes_str