"""
conftest.py — pytest configuration for reg-compliance-env.

Adds the project root to sys.path so tests can import modules directly.
Tells pytest to ignore the root __init__.py as a test file.
"""
import sys
import os

# Make root importable without installing the package
sys.path.insert(0, os.path.dirname(__file__))

collect_ignore = ["__init__.py", "setup.py"]
