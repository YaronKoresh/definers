class AnswerImageLoader:
    @staticmethod
    def read_answer_image(path: str, image_module):
        from definers.image import (
            get_max_resolution,
            image_resolution,
            resize_image,
        )

        try:
            return image_module.open(path)
        except Exception:
            try:
                shape = image_resolution(path)
                if (
                    isinstance(shape, tuple)
                    and len(shape) >= 2
                    and shape[0] > 0
                    and shape[1] > 0
                ):
                    width, height = shape[:2]
                    max_width, _max_height = get_max_resolution(
                        width,
                        height,
                        mega_pixels=0.25,
                    )
                    if max_width > width:
                        resized = resize_image(path, width, height)
                        if isinstance(resized, tuple):
                            return resized[1]
                        return resized
                    return image_module.open(path)
            except Exception:
                return None
        return None
