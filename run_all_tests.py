import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

loader = unittest.TestLoader()
suite = loader.discover(str(ROOT / "tests"))
runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=1)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
