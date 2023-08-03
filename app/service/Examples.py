import os
from pathlib import Path

from PIL.Image import Image


class Examples:

    selected_example_image = None

    def get_example_boxes(self, img_file, aspect_ratio=None):
        if self.selected_example_image == 'bingo1.PNG':
            return {'left': 130, 'top': 247, 'width': 63, 'height': 62}
        elif self.selected_example_image == 'bingo2.PNG':
            return {'left': 130, 'top': 373, 'width': 63, 'height': 62}
        elif self.selected_example_image == 'bingo3.PNG':
            return {'left': 8, 'top': 314, 'width': 55, 'height': 55}
        elif self.selected_example_image == 'bingo4.PNG':
            return {'left': 3, 'top': 309, 'width': 64, 'height': 64}
        elif self.selected_example_image == 'loading_screen.PNG':
            return {'left': 210, 'top': 235, 'width': 88, 'height': 36}
        elif self.selected_example_image == 'tournament1.PNG':
            return {'left': 246, 'top': 217, 'width': 103, 'height': 63}
        elif self.selected_example_image == 'tournament2.PNG':
            return {'left': 140, 'top': 217, 'width': 103, 'height': 63}
        elif self.selected_example_image == 'tournament3.PNG':
            return {'left': 352, 'top': 217, 'width': 103, 'height': 63}
        return {'left': img_file.width * .4, 'top': img_file.height * 0.4, 'width': img_file.width * .2, 'height': img_file.height * 0.2}

    def resized_width(img: Image, max_height: int = 700, max_width: int = 700) -> int:
        # Resize the image output of the result to the same width as the resized input img
        if img.height > img.width:
            ratio = max_height / img.height
        else:
            ratio = max_width / img.width
        return int(img.width * ratio)

    @staticmethod
    def get_resouce_path() -> Path:
        cwd = os.getcwd()
        return Path(cwd).joinpath("resources")

