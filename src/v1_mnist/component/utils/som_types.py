"""Type names for annotations of NNSOM objects.

NNSOM has three map classes with the same methods used here: SOM (NumPy),
SOMPlots (SOM plus plotting; what the project saves and loads) and SOMGpu
(CuPy). Annotations use SOMModel for any of them. Annotations are not checked
at run time, so importing this module does not import NNSOM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from NNSOM.plots import SOMPlots
    from NNSOM.som import SOM

SOMModel = Union["SOM", "SOMPlots"]
