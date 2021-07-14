from __future__ import annotations

from typing import Optional, Tuple, Union

Range = Union[int, Tuple[Optional[int], Optional[int]]]
Trim = Tuple[Optional[int], Optional[int]]


class DuplicateFrame(int):
    dup: int

    def __new__(cls, x: int, /, dup: int = 1) -> DuplicateFrame:
        fn = super().__new__(cls, x)
        fn.dup = dup
        return fn

    def __add__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup + x)

    def __sub__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup - x)

    def __mul__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup * x)

    def __floordiv__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup // x)
