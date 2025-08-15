import argparse
import csv
import datetime
import json
import math

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from numpy.typing import NDArray


def get_size(size_in_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_in_bytes)
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    size = math.trunc(size * 10**2) / 10**2
    return f'{size} {units[i]}'


@dataclass
class ImageFileManager:
    in_filename: Path | None = field(init=True, default=None)
    in_width: int = field(init=False, default=-1)
    in_height: int = field(init=False, default=-1)
    in_format: str | None = field(init=False, default=None)
    in_data: NDArray[np.uint8] = field(
        init=False, default_factory=lambda: np.array([], dtype=np.uint8))
    out_filename: Path | None = field(init=False, default=None)
    out_width: int = field(init=False, default=-1)
    out_height: int = field(init=False, default=-1)
    out_format: str | None = field(init=False, default=None)
    out_data: NDArray[np.uint8] = field(
        init=False, default_factory=lambda: np.array([], dtype=np.uint8))
    quality: int = field(init=False, default=100)
    compression: int = field(init=False, default=0)
    available: bool = field(init=False, default=True)

    def __post_init__(self):
        if not self.in_filename.exists():
            self.available = False
            return
        self.in_format = self.in_filename.suffix
        self.in_data = cv2.imread(self.in_filename)
        self.in_height, self.in_width = self.in_data.shape[:2]

    def record_output_info(self, data, filename, save=True, **kwargs) -> bool:
        if data is None or self.in_data is None:
            self.available = False
            return False
        self.out_height, self.out_width = data.shape[:2]
        self.out_data = data
        self.out_filename = filename
        self.out_format = self.out_filename.suffix
        if save:
            if self.out_format == '.jpg' or self.out_format == '.jpeg':
                quality = kwargs.get('jpg_quality', 100)
                self.available = cv2.imwrite(
                    self.out_filename, self.out_data,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif self.out_format == '.png':
                compression = kwargs.get('png_compression', 0)
                self.available = cv2.imwrite(
                    self.out_filename, self.out_data,
                    [cv2.IMWRITE_PNG_COMPRESSION, compression])
            else:
                self.available = cv2.imwrite(self.out_filename, self.out_data)
        return self.available

    def get_info(self):
        if self.available and self.out_filename is None:
            print(f'The output of {self.in_filename.name} may not exist.')
        info_dict = {}
        info_dict['input name'] = str(self.in_filename)
        info_dict['input width'] = self.in_width
        info_dict['input height'] = self.in_height
        info_dict['input size'] = get_size(self.in_filename.stat().st_size)
        info_dict['output name'] = str(self.out_filename)
        info_dict['output width'] = self.out_width
        info_dict['output height'] = self.out_height
        info_dict['output size'] = get_size(self.out_filename.stat().st_size)
        info_dict['available'] = self.available
        info_dict['datetime'] = str(datetime.datetime.now())
        return info_dict


@dataclass
class ImageResizeManager:
    fx: float | None = field(init=False, default=None)
    fy: float | None = field(init=False, default=None)
    width: int | None = field(init=False, default=None)
    height: int | None = field(init=False, default=None)
    keep_ratio: bool = field(init=True)
    set_params: bool = field(init=False)

    def __post_init__(self):
        self.check_size_and_scale()

    def check_size_and_scale(self) -> bool:
        check1 = sum(opt is not None for opt in [self.width, self.fx])
        check2 = sum(opt is not None for opt in [self.height, self.fy])
        check3 = 1 if self.keep_ratio else 0
        if ((check1 == 1 and check2 == 1 and check3 == 0) or
                (check1 == 1 and check2 == 0 and check3 == 1) or
                (check1 == 0 and check2 == 1 and check3 == 1)):
            self.set_params = True
            return True
        else:
            return False

    def set_size_or_scale(
                self,
                width: int | None = None,
                height: int | None = None,
                fx: float | None = None,
                fy: float | None = None
            ) -> bool:
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        return self.check_size_and_scale()

    def resize(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not self.set_params:
            raise RuntimeError('"set_size_or_scale()" should be called '
                               'before resizing an image.')

        h, w = img.shape[:2]
        if self.keep_ratio:
            if self.fx is not None:
                new_w = int(w * self.fx)
                new_h = int(h * self.fx)
            elif self.fy is not None:
                new_w = int(w * self.fy)
                new_h = int(h * self.fy)
            elif self.width is not None:
                new_w = self.width
                new_h = int(h * self.width / w)
            else:
                new_w = int(w * self.height / h)
                new_h = self.height
        elif self.fx is not None and self.fy is not None:
            new_w = int(w * self.fx)
            new_h = int(h * self.fx)
        elif self.width is not None and self.height is not None:
            new_w = self.width
            new_h = self.height
        elif self.width is not None and self.fy is not None:
            new_w = self.width
            new_h = int(h * self.fy)
        else:
            new_w = int(w * self.fx)
            new_h = self.height
        new_img = cv2.resize(img, (new_w, new_h))

        return new_img


class BatchResizer:
    def __init__(self,
                 in_dir: Path | str,
                 out_dir: Path | str,
                 out_format: str,
                 jpg_quality: int = 100,
                 png_compression: int = 0,
                 recuresive: bool = False,
                 savelog: str | None = None):
        self.img_extensions = ['.bmp', '.png', '.gif', '.jpeg', '.jpg']
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.out_dir = self.out_dir.relative_to(self.in_dir.parent)
        self.infos: list[ImageFileManager] = []
        self.resizer: ImageResizeManager = None
        self.savelog = None
        if savelog is not None:
            savelog = savelog.lower()
            if not savelog.startswith('.'):
                savelog = '.' + savelog
            if savelog not in ['.csv', '.json']:
                raise ValueError('Log format should be CSV or JSON format.')
            self.savelog = 'resize_log' + savelog

        if out_format is None:
            raise ValueError('Output format should be defined.')
        self.out_format = out_format
        if not out_format.startswith('.'):
            self.out_format = '.' + out_format
        if self.out_format not in self.img_extensions:
            raise ValueError(f'"{self.out_format}" is not supported.')

        self.jpg_quality = jpg_quality
        self.png_compression = png_compression

        if not in_dir.is_dir():
            raise TypeError(f'Error: Input directory not found: {self.in_dir}')
        for ifile in self.in_dir.glob('*/**' if recuresive else '*'):
            if ifile.suffix.lower() not in self.img_extensions:
                continue
            one_img_info = ImageFileManager(in_filename=ifile)
            self.infos.append(one_img_info)
        if len(self.infos) == 0:
            raise RuntimeError(f'No image found in {self.in_dir}.')
        print(f'Find {len(self.infos)} images in {self.in_dir}.')

    def get_min_width(self):
        return min([img.in_width for img in self.infos])

    def get_max_width(self):
        return max([img.in_width for img in self.infos])

    def get_min_height(self):
        return min([img.in_height for img in self.infos])

    def get_max_height(self):
        return max([img.in_height for img in self.infos])

    def set_resizer(self, resizer: ImageResizeManager):
        self.resizer = resizer

    def process_images(self):
        if self.resizer is None:
            raise RuntimeError('"ImageResizeManager" should be defined '
                               'before image processing.')
        if self.savelog is not None:
            print('save log')
            info_log = []
        for info in self.infos:
            filename = info.in_filename.stem + self.out_format
            pathname = self.out_dir / Path(*info.in_filename.parts[1:-1])
            pathname.mkdir(parents=True, exist_ok=True)
            out_data = self.resizer.resize(info.in_data)
            info.record_output_info(
                out_data, pathname / filename,
                jpg_quality=self.jpg_quality,
                png_compression=self.png_compression)
            if self.savelog is not None:
                info_log.append(info.get_info())

        if self.savelog is not None:
            if Path(self.savelog).suffix == '.json':
                with open(self.savelog, 'w', encoding='utf-8') as f:
                    json.dump(info_log, f, ensure_ascii=False, indent=4)
            elif Path(self.savelog).suffix == '.csv':
                with open(self.savelog, 'w', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=info_log[0].keys())
                    writer.writeheader()
                    writer.writerows(info_log)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch resize images with flexible options.'
    )

    # Input / Output
    parser.add_argument(
        '-i', '--input-dir', type=str, required=True,
        help='Input directory containing images.')
    parser.add_argument(
        '-o', '--output-dir', type=str,
        help='Output directory for resized images (default: overwrite input).')

    # Size or scale
    parser.add_argument(
        '-fx', type=float, default=None,
        help='Horizontal scale factor (e.g. 0.5 for half size).')
    parser.add_argument(
        '-fy', type=float, default=None,
        help='Vertical scale factor (e.g. 0.5 for half size).')
    parser.add_argument(
        '-W', '--width', type=int, metavar='W', default=None,
        help='Fixed output width in pixels.')
    parser.add_argument(
        '-H', '--height', type=int, metavar='H', default=None,
        help='Fixed output height in pixels.')
    parser.add_argument(
        '-fmw', '--fit-min-width', action='store_true',
        help='Fit to the smallest width among images in the directory.')
    parser.add_argument(
        '-fMw', '--fit-max-width', action='store_true',
        help='Fit to the largest width among images in the directory.')
    parser.add_argument(
        '-fmh', '--fit-min-height', action='store_true',
        help='Fit to the smallest height among images in the directory.')
    parser.add_argument(
        '-fMh', '--fit-max-height', action='store_true',
        help='Fit to the largest height among images in the directory.')

    # Aspect ratio
    parser.add_argument('-k', '--keep-ratio', action='store_true',
                        help='Keep aspect ratio when resizing.')

    # Output format
    parser.add_argument(
        '-f', '--format', type=str,
        help='Output image format (e.g. jpg, png).')
    parser.add_argument(
        '-q', '--quality', type=int,
        help='JPEG quality (1-100).', default=100)
    parser.add_argument(
        '-c', '--compression', type=int,
        help='PNG compression level (0-9).', default=0)

    # Other options
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help='Process subdirectories recursively.')
    parser.add_argument(
        '-l', '--log', type=str, default=None,
        help='log file extension [csv or json].')

    args = parser.parse_args()

    horizontal_opts1 = [args.width, args.fx]
    horizontal_opts2 = [args.fit_min_width, args.fit_max_width]
    check_horizontal = sum([opt is not None for opt in horizontal_opts1]
                           + horizontal_opts2)
    if check_horizontal > 1:
        raise ValueError('Cannot define more than one horizontal option.')
    vertical_opts1 = [args.height, args.fy]
    vertical_opts2 = [args.fit_min_height, args.fit_max_height]
    check_vertical = sum([opt is not None for opt in vertical_opts1]
                         + vertical_opts2)
    if check_vertical > 1:
        raise ValueError('Cannot define more than one vertical option.')
    if args.keep_ratio:
        if check_horizontal + check_vertical != 1:
            raise ValueError(
                'Horizontal and Vertical options cannot be defined '
                'when keep-ratio is set.')
    else:
        if check_horizontal + check_vertical != 2:
            raise ValueError(
                'Horizontal and Vertical options should be defined.')
    return args


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Start process...')
    batch_resizer = BatchResizer(
        input_dir, output_dir, args.format,
        jpg_quality=args.quality, png_compression=args.compression,
        recuresive=args.recursive, savelog=args.log)
    width = args.width
    if args.fit_min_width:
        width = batch_resizer.get_min_width()
    elif args.fit_max_width:
        width = batch_resizer.get_max_width()

    height = args.height
    if args.fit_min_height:
        height = batch_resizer.get_min_height()
    elif args.fit_max_height:
        height = batch_resizer.get_max_height()        

    img_resize_manager = ImageResizeManager(keep_ratio=args.keep_ratio)
    img_resize_manager.set_size_or_scale(width=width, height=height,
                                         fx=args.fx, fy=args.fy)
    batch_resizer.set_resizer(img_resize_manager)
    batch_resizer.process_images()


if __name__ == '__main__':
    main()
