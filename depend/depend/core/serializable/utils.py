from pandas import DataFrame
import pyarrow as pa
import jsonpickle


def serialize_with_pyarrow(dataframe: DataFrame):
    batch = pa.record_batch(dataframe)
    write_options = pa.ipc.IpcWriteOptions(compression="zstd")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema,   options=write_options) as writer:
        writer.write_batch(batch)
    pybytes = sink.getvalue().to_pybytes()
    pybytes_str = jsonpickle.encode(pybytes, unpicklable=True, make_refs=False)
    return pybytes_str

 