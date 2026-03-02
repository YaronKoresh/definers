from definers._system import catch, tmp

try:
    import cupy as np
except Exception:
    import numpy as np

import numpy as _np


def extract_video_features(video_path, frame_interval=10):
    import cv2
    import skimage.feature as skf

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
                hist_b = cv2.calcHist(
                    [frame], [0], None, [256], [0, 256]
                ).flatten()
                hist_g = cv2.calcHist(
                    [frame], [1], None, [256], [0, 256]
                ).flatten()
                hist_r = cv2.calcHist(
                    [frame], [2], None, [256], [0, 256]
                ).flatten()
                color_hist = _np.concatenate((hist_b, hist_g, hist_r)).astype(
                    _np.float32
                )
                radius = 1
                n_points = 8 * radius
                lbp = (
                    skf.local_binary_pattern(
                        frame_gray, n_points, radius, method="uniform"
                    )
                    .flatten()
                    .astype(_np.float32)
                )
                edges = (
                    cv2.Canny(frame_gray, 100, 200)
                    .flatten()
                    .astype(_np.float32)
                )
                frame_features = _np.concatenate((color_hist, lbp, edges))
                all_frame_features.append(frame_features)
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
    import definers as _d
    import cv2

    if predicted_features is None or predicted_features.size == 0:
        return False
    output_path = _d.tmp("mp4")
    try:
        (height, width, channels) = video_shape
        hist_size = 256 * 3
        lbp_size = height * width
        height * width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if predicted_features.ndim == 1:
            predicted_features = predicted_features.reshape(1, -1)
        for frame_features in predicted_features:
            color_hist = frame_features[:hist_size].reshape(3, 256)
            lbp_features = frame_features[
                hist_size : hist_size + lbp_size
            ].reshape(height, width)
            edge_features = frame_features[hist_size + lbp_size :].reshape(
                height, width
            )
            reconstructed_frame = np.zeros(video_shape, dtype=np.uint8)
            for c in range(channels):
                for i in range(256):
                    if c == 0:
                        reconstructed_frame[:, :, 0] += np.uint8(
                            color_hist[0][i] / np.max(color_hist[0]) * 255
                        )
                    elif c == 1:
                        reconstructed_frame[:, :, 1] += np.uint8(
                            color_hist[1][i] / np.max(color_hist[1]) * 255
                        )
                    else:
                        reconstructed_frame[:, :, 2] += np.uint8(
                            color_hist[2][i] / np.max(color_hist[2]) * 255
                        )
            lbp_scaled = cv2.normalize(
                lbp_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            edge_scaled = cv2.normalize(
                edge_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            reconstructed_frame_gray = cv2.addWeighted(
                lbp_scaled, 0.5, edge_scaled, 0.5, 0
            )
            reconstructed_frame = cv2.cvtColor(
                reconstructed_frame, cv2.COLOR_BGR2GRAY
            )
            reconstructed_frame = cv2.addWeighted(
                reconstructed_frame, 0.5, reconstructed_frame_gray, 0.5, 0
            )
            reconstructed_frame = cv2.cvtColor(
                reconstructed_frame, cv2.COLOR_GRAY2BGR
            )
            out.write(reconstructed_frame)
        out.release()
        return output_path
    except Exception as e:
        print(f"Error generating video from features: {e}")
        return False


def resize_video(
    input_video_path, target_height, target_width, anti_aliasing=True
):
    import definers as _d
    import imageio as iio
    from skimage.transform import resize

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
    import definers as _d
    import imageio as iio

    output_video_path = _d.tmp("mp4")
    try:
        reader = iio.imiter(input_video_path)
        metadata = reader.metadata()
        original_fps = metadata["fps"]
        frames = list(reader)
        reader.close()
        if original_fps == target_fps:
            iio.imwrite(output_video_path, frames, fps=target_fps)
            return
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
    import definers as _d
    import imageio as iio

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
