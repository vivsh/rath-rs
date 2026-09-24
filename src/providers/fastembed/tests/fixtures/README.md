# Native inference fixture

These files are original Rath test fixtures licensed under the repository's
MIT OR Apache-2.0 license. `generate.py` regenerates them without external
packages. The ONNX graph gathers deterministic, synthetic 384-dimensional
vectors from a four-token table. It tests ONNX loading, tokenization, pooling,
batching and synchronization. It is not trained and provides no evidence of
real-model embedding quality. No upstream model weights are included.
