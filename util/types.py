from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from PIL.Image import Image as PILImage


@dataclass
class Resolution:
    w: int
    h: int


    @classmethod
    def from_image(cls, image: PILImage) -> Resolution:
        """
        Takes in an Image object and returns a Resolution with w and h identical to the image's w
        and h

        Args:
            image (Image): Input image
        
        Returns:
            Resolution: The input image's resolution
        """
        w, h = image.size
        return Resolution(w=w, h=h)


    def as_hw_tuple(self) -> Tuple[int, int]:
        return (self.h, self.w)


    def as_wh_tuple(self) -> Tuple[int, int]:
        return (self.w, self.h)


    def pad(self, padding: List[int]):
        """
        Applies padding

        Args:
            padding (List[int]): Padding for the left, top, right and bottom borders respectively
        """
        return Resolution(
            w=self.w + padding[0] + padding[2],
            h=self.h + padding[1] + padding[3],
        )
        