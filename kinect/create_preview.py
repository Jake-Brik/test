import os
from sys import platform
from ctypes import c_wchar_p, windll
from ctypes.wintypes import DWORD

if platform.lower().startswith('win'):
    # This code can fix possible error between windows 10 and Azure Kinect k4a.dll
    k4a_dll_path = 'C:/Program Files/Azure Kinect SDK v1.4.1/sdk/windows-desktop/amd64/release/bin'
    check_file = os.path.exists(k4a_dll_path)
    if check_file:
        AddDllDirectory = windll.kernel32.AddDllDirectory
        AddDllDirectory.restype = DWORD
        AddDllDirectory.argtypes = [c_wchar_p]
        AddDllDirectory(k4a_dll_path)
    else:
        print(f'Error: You need to have k4a.dll at {k4a_dll_path}')

import cv2
import numpy as np
from typing import Optional, Tuple
from pyk4a import ImageFormat, PyK4APlayback, depth_image_to_color_camera, color_image_to_depth_camera
from argparse import ArgumentParser


def colorize(
        image: np.ndarray,
        clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
        colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img = clahe.apply(img)
    img = cv2.applyColorMap(img, colormap)
    return img


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    if color_format == ImageFormat.COLOR_MJPG:
        rgb_image = cv2.imdecode(color_image, cv2.IMREAD_UNCHANGED)
        color_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def main():
    # Number of previews you want to get
    NUMBER = 10
    # You can set up the frame for the first preview.
    # For correct work set up the first preview forward at least to 0.07 sec
    CURRENTFRAME = 70000
    parser = ArgumentParser(description='Create previews from video')
    parser.add_argument('--channel', action="store", choices=['color', 'depth', 'ir'],
                        type=str, default='color', help='Set channel you want to use')
    parser.add_argument('--transform', action="store", choices=['depth', 'color'], type=str, default=None,
                        help='You can transform depth to color space, or color to dept space')
    parser.add_argument('path', action="store", type=str,
                        help='Read the video from the specified path')
    parser.add_argument('out', action="store", type=str, help='Set output path for previews')

    args = parser.parse_args()
    transform = args.transform
    channel = args.channel
    path = args.path
    out = args.out

    # Read and open file
    playback = PyK4APlayback(path)
    playback.open()
    # An evenly spaced sequence of frames to create previews on specific points
    points = np.linspace(CURRENTFRAME, playback.length, NUMBER, dtype=int)

    try:
        # Creating a folder named data
        if not os.path.exists(out):
            os.makedirs(out)
        # If not created and not exist then raise an error
    except OSError as err:
        print("OS error: {0}".format(err))

    for point in points:
        playback.seek(int(point))
        captured = playback.get_previouse_capture()
        picture = convert_to_bgra_if_required(playback.configuration["color_format"], captured.color)

        if transform and channel != 'ir':
            name = f'{channel}_transformed {point / 1000000:.0f}.jpg'
        else:
            name = f'{channel} {point / 1000000:.0f}.jpg'

        if channel == 'color' and captured.color is not None:
            if transform == 'depth':
                picture = color_image_to_depth_camera(picture, captured.depth, playback.calibration, True)
        elif channel == 'depth' and captured.depth is not None:
            if transform == 'color':
                transformed_depth = depth_image_to_color_camera(captured.depth, playback.calibration, True)
                picture = colorize(transformed_depth, (None, 4000))
            else:
                picture = colorize(captured.depth, (None, 5000))
        elif channel == 'ir' and captured.ir is not None:
            picture = colorize(captured.ir, (None, 500), colormap=cv2.COLORMAP_JET)

        print(f'Creating..{out}/{name}')
        cv2.imwrite(f'{out}/{name}', picture)


if __name__ == "__main__":
    main()
