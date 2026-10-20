
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.mri_preprocessing import preprocess_mri


class TestMRIProcessing(unittest.TestCase):

    def test_preprocess_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            input_path = root / "sample.png"
            output_path = root / "processed" / "sample.png"

            sample = np.zeros((128, 128), dtype=np.uint8)
            sample[40:80, 40:80] = 180

            Image.fromarray(sample).save(input_path)

            result = preprocess_mri(
                str(input_path),
                str(output_path),
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(result["shape"], (256, 256))
            self.assertGreaterEqual(result["minimum"], 0.0)
            self.assertLessEqual(result["maximum"], 1.0)


if __name__ == "__main__":
    unittest.main()
