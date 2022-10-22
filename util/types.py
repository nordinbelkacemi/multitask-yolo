from typing import List, NamedTuple
from PIL import Image


class Resolution(NamedTuple):
    w: int
    h: int


    @classmethod
    def from_image(cls, image: Image):
        """
        Takes in an Image object and returns a Resolution with w and h identical to the image's w
        and h
        """
        w, h = image.size
        return Resolution(w=w, h=h)


    def pad(self, padding: List[int]):
        """
        Applies padding

        Args:
            - padding (List[int]): Padding for the left, top, right and bottom borders respectively
        """
        return Resolution(
            w=self.w + padding[0] + padding[2],
            h=self.h + padding[1] + padding[3],
        )