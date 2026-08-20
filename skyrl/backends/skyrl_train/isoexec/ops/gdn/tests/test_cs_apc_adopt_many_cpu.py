"""CPU contracts for scheduler-wave CS-APC checkpoint adoption."""

from __future__ import annotations

from collections import OrderedDict

import torch

from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_chunk_synced_state as cs


class _Layer:
    def __init__(self, layer: int, capacity: int = 8):
        self.layer = layer
        self.chunk_size = 64
        self.capacity = capacity
        self.ssm_state = torch.zeros(capacity + 1, 1, 2, 2, dtype=torch.float32)
        self.conv_state = torch.zeros(capacity + 1, 3, 2, dtype=torch.bfloat16)
        self.entry_state = torch.zeros_like(self.ssm_state)
        self.pos = torch.zeros(capacity + 1, dtype=torch.long)
        self._entry_host = False
        self._apc_pending = {r: [(64, None, None)] for r in range(1, capacity + 1)}
        self._row_pos = {}
        self._entry_pos = {}
        self._slot2row = OrderedDict()
        self._free = list(range(1, capacity + 1))
        self.assign_calls = 0

    def _assign_many(self, slots):
        self.assign_calls += 1
        rows = []
        for slot in slots:
            row = self._slot2row.get(int(slot))
            if row is None:
                row = self._free.pop()
                self._slot2row[int(slot)] = row
            rows.append(row)
        return rows


def _install_store(monkeypatch, layers):
    store = cs.CSBoundaryStore(1 << 20)
    entries = {
        b"a": torch.tensor(
            [
                [[[1.001, -2.003], [3.007, -4.011]]],
                [[[5.013, -6.017], [7.019, -8.023]]],
            ],
            dtype=torch.float32,
        ),
        b"b": torch.tensor(
            [
                [[[9.029, -10.031], [11.037, -12.041]]],
                [[[13.043, -14.047], [15.053, -16.057]]],
            ],
            dtype=torch.float32,
        ),
    }
    convs = {
        key: torch.arange(len(layers) * 6, dtype=torch.bfloat16).reshape(len(layers), 3, 2)
        + offset
        for key, offset in ((b"a", 0), (b"b", 20))
    }
    assert store.put(b"a", 64, entries[b"a"], convs[b"a"])
    assert store.put(b"b", 128, entries[b"b"], convs[b"b"])
    monkeypatch.setattr(cs, "CS_APC_STORE", store)
    return entries, convs


def test_adopt_many_batches_assignment_and_preserves_exact_state(monkeypatch):
    layers = [_Layer(0), _Layer(1)]
    entries, convs = _install_store(monkeypatch, layers)
    items = [(101, 64, b"a"), (102, 128, b"b"), (103, 64, b"a")]

    rows = cs.cs_apc_adopt_many(layers, items)

    assert rows == [8, 7, 6]
    assert [layer.assign_calls for layer in layers] == [1, 1]
    for j, layer in enumerate(layers):
        expected_entry = torch.stack([entries[key][j] for _slot, _pos, key in items])
        expected_conv = torch.stack([convs[key][j] for _slot, _pos, key in items])
        assert torch.equal(layer.entry_state[rows], expected_entry)
        assert torch.equal(
            layer.ssm_state[rows], expected_entry.to(torch.bfloat16).to(torch.float32)
        )
        assert torch.equal(layer.conv_state[rows], expected_conv)
        assert torch.equal(layer.pos[rows], torch.tensor([64, 128, 64]))
        assert [layer._row_pos[row] for row in rows] == [64, 128, 64]
        assert [layer._entry_pos[row] for row in rows] == [64, 128, 64]
        assert all(row not in layer._apc_pending for row in rows)


def test_adopt_many_miss_is_atomic(monkeypatch):
    layers = [_Layer(0), _Layer(1)]
    _install_store(monkeypatch, layers)

    rows = cs.cs_apc_adopt_many(layers, [(101, 64, b"a"), (102, 128, b"missing")])

    assert rows is None
    assert [layer.assign_calls for layer in layers] == [0, 0]
    assert all(not layer._slot2row for layer in layers)
    assert all(torch.count_nonzero(layer.ssm_state) == 0 for layer in layers)


def test_scalar_door_keeps_refusal_contract(monkeypatch):
    layers = [_Layer(0), _Layer(1)]
    _install_store(monkeypatch, layers)

    assert cs.cs_apc_adopt(layers, 101, 64, b"a")
    assert not cs.cs_apc_adopt(layers, 102, 65, b"a")
    assert not cs.cs_apc_adopt(layers, 103, 64, b"missing")
