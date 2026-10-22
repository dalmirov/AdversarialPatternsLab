import json
import threading
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

from app import PatternLabHandler, PatternLabServer

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


class PatternLabApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = PatternLabServer(("127.0.0.1", 0), PatternLabHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def test_catalog_endpoint(self):
        with urllib.request.urlopen(self.base_url + "/api/catalog", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        self.assertEqual(payload["count"], len(REGISTRY))
        self.assertEqual(len(payload["patterns"]), len(REGISTRY))

    def test_pattern_png_endpoint(self):
        url = self.base_url + "/api/pattern.png?name=blocks&seed=2&size=96&tile_size=64"
        with urllib.request.urlopen(url, timeout=5) as response:
            png = response.read()
            self.assertEqual(response.headers.get("Content-Type"), "image/png")
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (64, 64, 3))

    def test_invalid_pattern_returns_bad_request(self):
        url = self.base_url + "/api/pattern.png?name=not-real&seed=2&size=96&tile_size=64"
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(url, timeout=5)
        self.assertEqual(caught.exception.code, 400)

    def test_homepage_is_served(self):
        with urllib.request.urlopen(self.base_url + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
        self.assertIn("PATTERN LAB", html)
        self.assertIn("/api/catalog", html)


if __name__ == "__main__":
    unittest.main()
