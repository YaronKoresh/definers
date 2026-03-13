import os
import shutil
import tempfile
import unittest
import zipfile

from definers._system import modify_wheel_requirements


class TestModifyWheelRequirements(unittest.TestCase):
    def make_wheel(self, metadata_lines):

        tempdir = tempfile.mkdtemp()
        distinfo = os.path.join(tempdir, "pkg-1.0.dist-info")
        os.makedirs(distinfo)
        metadata_path = os.path.join(distinfo, "METADATA")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))
        wheel_path = os.path.join(tempdir, "pkg-1.0-py3-none-any.whl")
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(tempdir):
                for file in files:
                    full = os.path.join(root, file)
                    arc = os.path.relpath(full, tempdir)
                    z.write(full, arc)
        return wheel_path, tempdir

    def read_metadata(self, wheel_path):
        with zipfile.ZipFile(wheel_path, "r") as z:
            names = [n for n in z.namelist() if n.endswith("METADATA")]
            self.assertTrue(names)
            with z.open(names[0]) as f:
                return f.read().decode("utf-8")

    def test_add_and_modify(self):
        lines = [
            "Metadata-Version: 2.1",
            "Name: pkg",
            "Requires-Dist: foo (>=1.0)",
        ]
        wheel, tmpdir = self.make_wheel(lines)
        try:
            new_wheel = modify_wheel_requirements(
                wheel, {"foo": "2.0", "bar": "1.0"}
            )
            content = self.read_metadata(new_wheel)
            self.assertIn("Requires-Dist: foo (2.0)", content)
            self.assertIn("Requires-Dist: bar (1.0)", content)
        finally:
            shutil.rmtree(tmpdir)

    def test_remove_dependency(self):
        lines = [
            "Metadata-Version: 2.1",
            "Name: pkg",
            "Requires-Dist: baz (>=0.1)",
        ]
        wheel, tmpdir = self.make_wheel(lines)
        try:
            new_wheel = modify_wheel_requirements(wheel, {"baz": None})
            content = self.read_metadata(new_wheel)
            self.assertNotIn("baz", content)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
