"""SO-RECON: physically constrained Bayesian reconstruction of current oil saturation."""

from typing import Literal, get_args

__version__ = "0.0.1"

#: The specification versions this codebase reads and writes. E00 records stay at 3.0;
#: everything produced from E01 onwards is 4.0 (SPEC 0, 17.4.1). The tuple is derived from
#: the type so that the runtime list and the validated Literal can never disagree.
SpecVersion = Literal["3.0", "4.0"]
SUPPORTED_SPEC_VERSIONS: tuple[SpecVersion, ...] = get_args(SpecVersion)
LATEST_SPEC_VERSION: SpecVersion = "4.0"

#: Legacy alias for the version E00 stamped from this module. New code takes the version
#: from the validated configuration instead; this stays while its consumers exist.
SPEC_VERSION = "3.0"
