import importlib.util
import pathlib
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).parents[2] / "Assets" / "Models" / "model_metadata_generator.py"
SPEC = importlib.util.spec_from_file_location("model_metadata", SCRIPT)
model_metadata = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model_metadata
SPEC.loader.exec_module(model_metadata)


class ModelMetadataTests(unittest.TestCase):
    def test_closed_tetrahedron_metadata(self):
        obj = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tetra.obj"
            path.write_text(obj, encoding="utf-8")
            metadata = model_metadata.analyze_obj(path)

        self.assertEqual(set(metadata), {
            "vertexCount",
            "triangleCount",
            "sourceExtent",
            "enclosedVolume",
            "watertight",
        })
        self.assertEqual(metadata["vertexCount"], 4)
        self.assertEqual(metadata["triangleCount"], 4)
        self.assertEqual(metadata["sourceExtent"], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(metadata["enclosedVolume"], 1.0 / 6.0)
        self.assertTrue(metadata["watertight"])

    def test_open_quad_uses_fan_triangulation_and_null_volume(self):
        obj = """v 0 0 0
v 2 0 0
v 2 1 0
v 0 1 0
f 1 2 3 4
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "quad.obj"
            path.write_text(obj, encoding="utf-8")
            metadata = model_metadata.analyze_obj(path)

        self.assertEqual(set(metadata), {
            "vertexCount",
            "triangleCount",
            "sourceExtent",
            "enclosedVolume",
            "watertight",
        })
        self.assertEqual(metadata["vertexCount"], 4)
        self.assertEqual(metadata["triangleCount"], 2)
        self.assertEqual(metadata["sourceExtent"], [2.0, 1.0, 0.0])
        self.assertIsNone(metadata["enclosedVolume"])
        self.assertFalse(metadata["watertight"])


    def test_serialization_keeps_three_value_vectors_on_one_line(self):
        catalog = {
            "models": {
                "Example": {
                    "sourceToCanonical": {"uDir": [1.0, 0.0, -1.0]},
                    "metadata": {"sourceExtent": [2.0, 3.5, 4.0]},
                }
            }
        }

        serialized = model_metadata.serialize_catalog(catalog)

        self.assertIn('"uDir": [1.0, 0.0, -1.0]', serialized)
        self.assertIn('"sourceExtent": [2.0, 3.5, 4.0]', serialized)
        self.assertNotIn('"uDir": [\n', serialized)
        self.assertNotIn('"sourceExtent": [\n', serialized)

if __name__ == "__main__":
    unittest.main()
