"""Canonical serialization: byte stability and strictness."""

import json
import unittest

from skyrl.backends.skyrl_train.isoexec.contract import (
    SerializationError,
    UnknownFieldError,
    from_canonical_json,
    to_canonical_json,
)
from skyrl.backends.skyrl_train.isoexec.contract.tests.fixtures import base_contract


class TestRoundTrip(unittest.TestCase):
    def test_byte_stable(self):
        c = base_contract()
        b = to_canonical_json(c)
        self.assertEqual(to_canonical_json(from_canonical_json(b)), b)

    def test_decode_equals_original(self):
        c = base_contract()
        self.assertEqual(from_canonical_json(to_canonical_json(c)), c)

    def test_unknown_field_refused(self):
        d = json.loads(to_canonical_json(base_contract()))
        d["composition"][0]["surprise"] = 1
        with self.assertRaises(UnknownFieldError):
            from_canonical_json(json.dumps(d).encode())

    def test_unknown_toplevel_field_refused(self):
        d = json.loads(to_canonical_json(base_contract()))
        d["extra"] = {}
        with self.assertRaises(UnknownFieldError):
            from_canonical_json(json.dumps(d).encode())

    def test_float_leak_refused_on_decode(self):
        d = json.loads(to_canonical_json(base_contract()))
        d["composition"][0]["constants"]["bad"] = 0.5
        with self.assertRaises(SerializationError):
            from_canonical_json(json.dumps(d).encode())

    def test_float_leak_refused_on_encode(self):
        import dataclasses

        c = base_contract()
        e = dataclasses.replace(c.composition[0], constants={"bad": 0.5})
        c2 = dataclasses.replace(c, composition=(e,) + c.composition[1:])
        with self.assertRaises(SerializationError):
            to_canonical_json(c2)

    def test_legacy_evidence_field_refused(self):
        # evidence was removed from the schema; old artifacts carrying it must refuse to load.
        d = json.loads(to_canonical_json(base_contract()))
        d["composition"][0]["evidence"] = ["gates/old_pointer"]
        with self.assertRaises(UnknownFieldError):
            from_canonical_json(json.dumps(d).encode())

    def test_schema_version_refused(self):
        d = json.loads(to_canonical_json(base_contract()))
        d["schema_version"] = "99"
        with self.assertRaises(SerializationError):
            from_canonical_json(json.dumps(d).encode())

    def test_superseded_schema_version_refused(self):
        # A "1" artifact hashed its cases by id alone: it must refuse, not re-hash under these rules.
        d = json.loads(to_canonical_json(base_contract()))
        d["schema_version"] = "1"
        with self.assertRaises(SerializationError) as cm:
            from_canonical_json(json.dumps(d).encode())
        # ...and name the versions it does read, plus how to get one.
        msg = str(cm.exception)
        self.assertIn("['2']", msg)
        self.assertIn("rebuild the contract", msg)


if __name__ == "__main__":
    unittest.main()
