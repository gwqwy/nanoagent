import sys
import unittest

sys.path.insert(0, r"E:\文件\编程文件\agent\nanoagent")

loader = unittest.TestLoader()
suite = loader.discover(r"E:\文件\编程文件\agent\nanoagent\tests")
runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=1)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
