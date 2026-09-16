"""SO-RECON E03: the learned inverse loop package.

Importing this package imports nothing heavy: Torch and nflows are imported inside the
modules that use them (`ml.flow`, `ml.train`, `ml.encoders`) so config validation and
contract tests never pay for a backend they do not touch.
"""
