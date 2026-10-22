import unittest

import numpy as np

from pattern_library import REGISTRY
from pattern_lab import DEFAULT_SOURCE_SIZE, generate_patch, get_catalog, repeated_preview, variant_metadata


class PatternLabTests(unittest.TestCase):
    def test_catalog_exposes_all_existing_patterns(self):
        catalog = get_catalog()
        self.assertEqual(catalog["count"], len(REGISTRY))
        self.assertEqual(len(catalog["patterns"]), len(REGISTRY))
        self.assertIn("blocks", {item["name"] for item in catalog["patterns"]})

    def test_seed_is_reproducible(self):
        first = generate_patch("blocks", seed=57, size=96)
        second = generate_patch("blocks", seed=57, size=96)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (96, 96, 3))
        self.assertEqual(first.dtype, np.uint8)

    def test_all_registered_patterns_generate_valid_images(self):
        for name in REGISTRY:
            with self.subTest(pattern=name):
                image = generate_patch(name, seed=7, size=96)
                self.assertEqual(image.shape, (96, 96, 3))
                self.assertEqual(image.dtype, np.uint8)

    def test_repeated_preview_dimensions(self):
        patch = generate_patch("checker", seed=9, size=96)
        preview = repeated_preview(patch, tile_size=48, edge=240)
        self.assertEqual(preview.shape, (240, 240, 3))
        self.assertEqual(preview.dtype, np.uint8)

    def test_metadata_contains_reproduction_details(self):
        metadata = variant_metadata("stripes", seed=21, size=DEFAULT_SOURCE_SIZE, tile_size=150)
        self.assertEqual(metadata["pattern"], "stripes")
        self.assertEqual(metadata["seed"], 21)
        self.assertEqual(metadata["source_size_px"], DEFAULT_SOURCE_SIZE)
        self.assertEqual(metadata["display_tile_size_px"], 150)
        self.assertEqual(len(metadata["sha256_pixels_bgr"]), 64)
        self.assertEqual(metadata["license"], "GPL-3.0")

    def test_invalid_pattern_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_patch("not-a-pattern", seed=0, size=96)

    def test_invalid_seed_and_size_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_patch("blocks", seed=-1, size=96)
        with self.assertRaises(ValueError):
            generate_patch("blocks", seed=0, size=12)


if __name__ == "__main__":
    unittest.main()
