# Batch Image Resizer

## Description
This Python script resizes images in a directory with flexible options:
- Resize by absolute width/height, scale factor, or fit to min/max width/height.
- Keep aspect ratio if desired.
- Save images in different formats (JPEG, PNG) with quality/compression settings.
- Recursive processing of subdirectories.
- Log results in CSV or JSON format.

## Usage
```bash
python batch_resize.py -i <input_dir> -o <output_dir> -f jpg -W 800 -H 600 -k -r -l json
```

## Options
| Option                 | description                          |
|------------------------|--------------------------------------|
| -i, --input-dir        | Input directory containing images    |
| -o, --output-dir       | Output directory for resized images  |
| -fx                    | Horizontal scale factor              |
| -fy                    | Vertical scale factor                |
| -W, --width            | Fixed output width in pixels         |
| -H, --height           | Fixed output height in pixels        |
| -fmw, --fit-min-width  | Fit to smallest width among images   |
| -fmh, --fit-min-height | Fit to smallest height among images  |
| -fMw, --fit-max-width  | Fit to largest width among images    |
| -fMh, --fit-max-height | Fit to largest height among images   |
| -k, --keep-ratio       | Keep aspect ratio when resizing      |
| -f, --format           | Output image format (jpg, png, etc.) |
| -q, --quality          | JPEG quality (1-100)                 |
| -c, --compression      | PNG compression level (0-9)          |
| -r, --recursive        | Process subdirectories recursively   |
| -l, --log              | Save processing log as CSV or JSON   |