"""Generate an original tiny ONNX Gather fixture, using only Python's standard library.
SPDX-License-Identifier: MIT OR Apache-2.0
This is a protocol/runtime fixture, not a trained language embedding model.
"""
from pathlib import Path
import struct
import json

def varint(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)

def integer(field, value):
    return varint(field << 3) + varint(value)

def data(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint(field << 3 | 2) + varint(len(value)) + value

def tensor_type(name, element, dimensions):
    shape = b''.join(data(1, integer(1, d) if isinstance(d, int) else data(2, d)) for d in dimensions)
    return data(1, name) + data(2, data(1, integer(1, element) + data(2, shape)))

root = Path(__file__).parent
weights = b''.join(struct.pack('<f', ((row + column) % 17 + 1) / 17) for row in range(4) for column in range(384))
initializer = integer(1, 4) + integer(1, 384) + integer(2, 1) + data(8, 'weights') + data(9, weights)
node = data(1, 'weights') + data(1, 'input_ids') + data(2, 'last_hidden_state') + data(4, 'Gather')
graph = data(1, node) + data(2, 'rath_fixture') + data(5, initializer)
for name in ['input_ids', 'attention_mask', 'token_type_ids']:
    graph += data(11, tensor_type(name, 7, ['batch', 'sequence']))
graph += data(12, tensor_type('last_hidden_state', 1, ['batch', 'sequence', 384]))
model = integer(1, 8) + data(2, 'rath-test-fixture') + data(7, graph) + data(8, integer(2, 13))
(root / 'model.onnx').write_bytes(model)
tokenizer = {'version':'1.0', 'truncation':None, 'padding':None, 'added_tokens':[], 'normalizer':None, 'pre_tokenizer':{'type':'Whitespace'}, 'post_processor':None, 'decoder':None, 'model':{'type':'WordLevel','vocab':{'[PAD]':0,'[UNK]':1,'hello':2,'world':3},'unk_token':'[UNK]'}}
for name, value in {'tokenizer.json':tokenizer,'config.json':{'pad_token_id':0},'tokenizer_config.json':{'model_max_length':512,'pad_token':'[PAD]'},'special_tokens_map.json':{'unk_token':'[UNK]','pad_token':'[PAD]'}}.items():
    (root/name).write_text(json.dumps(value))
