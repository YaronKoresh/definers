from definers.data.runtime_patches import init_cupy_numpy
from definers.image.helpers import (
    _extract_visual_features,
    _reconstruct_visual_frame,
)
from definers.system import catch

np, _np = init_cupy_numpy()


def extract_video_features(video_path, frame_interval=10):
    import cv2

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file.")
        frame_count = 0
        all_frame_features = []
        while True:
            (ret, frame) = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_frame_features.append(
                    _extract_visual_features(frame, frame_gray)
                )
            frame_count += 1
        if not all_frame_features:
            return None
        return np.array(all_frame_features)
    except Exception as e:
        catch(e)
        return None
    finally:
        if cap is not None:
            cap.release()


def features_to_video(
    predicted_features, frame_interval=10, fps=24, video_shape=(1024, 1024, 3)
):
    import cv2

    import definers as _d

    if predicted_features is None or predicted_features.size == 0:
        return False
    output_path = _d.tmp("mp4")
    try:
        (height, width, channels) = video_shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if predicted_features.ndim == 1:
            predicted_features = predicted_features.reshape(1, -1)
        for frame_features in predicted_features:
            out.write(_reconstruct_visual_frame(frame_features, video_shape))
        out.release()
        return output_path
    except Exception as e:
        print(f"Error generating video from features: {e}")
        return False


def resize_video(
    input_video_path, target_height, target_width, anti_aliasing=True
):
    import imageio as iio
    from skimage.transform import resize

    import definers as _d

    output_video_path = _d.tmp("mp4")
    try:
        reader = iio.imiter(input_video_path)
        metadata = reader.metadata()
        fps = metadata["fps"]
        writer = iio.imwriter(output_video_path, fps=fps)
        for frame in reader:
            resized_frame = resize(
                frame,
                (target_height, target_width),
                anti_aliasing=anti_aliasing,
            )
            writer.append_data((resized_frame * 255).astype(np.uint8))
        writer.close()
        reader.close()
        return output_video_path
    except FileNotFoundError:
        print(f"Error: Video file not found at {input_video_path}")
    except Exception as e:
        print(f"An error occurred during video resizing: {e}")


def convert_video_fps(input_video_path, target_fps):
    import imageio as iio

    import definers as _d

    output_video_path = _d.tmp("mp4")
    try:
        reader = iio.imiter(input_video_path)
        metadata = reader.metadata()
        original_fps = metadata["fps"]
        frames = list(reader)
        reader.close()
        if original_fps == target_fps:
            iio.imwrite(output_video_path, frames, fps=target_fps)
            return output_video_path
        ratio = target_fps / original_fps
        new_frames = []
        for i in np.arange(0, len(frames), 1 / ratio):
            index = int(i)
            if index < len(frames):
                new_frames.append(frames[index])
        iio.imwrite(output_video_path, new_frames, fps=target_fps)
        return output_video_path
    except FileNotFoundError:
        print(f"Error: Video file not found at {input_video_path}")
    except Exception as e:
        print(f"An error occurred during 24 conversion: {e}")


def write_video(video_data, fps):
    import imageio as iio

    import definers as _d

    output_path = _d.tmp("mp4")
    try:
        writer = iio.imwriter(output_path, fps=fps)
        for frame in video_data:
            writer.append_data(frame)
        writer.close()
        return output_path
    except Exception as e:
        print(f"An error occurred during video writing: {e}")


def read_video(video_path):
    import imageio as iio

    try:
        reader = iio.imiter(video_path)
        metadata = reader.metadata()
        video_data = list(reader)
        reader.close()
        return (metadata, video_data)
    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}")
        return (None, None)
    except Exception as e:
        print(f"An error occurred during video reading: {e}")
        return (None, None)
