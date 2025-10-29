"""
Check image pixel ranges and optionally fix improper dtypes/ranges.

Usage:
    python scripts/check_image_ranges.py [--fix]

Checks:
 - Images should be uint8 with max <= 255 and min >= 0
 - If float images are found, reports their ranges. With --fix, converts float in 0-1 -> uint8 by *255 and clips.
"""

import sys
import glob
import os
import cv2
import numpy as np

FIX = '--fix' in sys.argv


def find_images(data_dir='data'):
    patterns = [os.path.join(data_dir, '**', '*.jpg'), os.path.join(data_dir, '**', '*.png')]
    imgs = []
    for p in patterns:
        imgs.extend(glob.glob(p, recursive=True))
    return imgs


def check_and_fix():
    imgs = find_images()
    if not imgs:
        print('>>>>> No images found under data/.')
        return 1

    issues = []
    for p in imgs:
        try:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                issues.append((p, 'cannot_read'))
                continue
            minv = float(np.min(im))
            maxv = float(np.max(im))
            dtype = im.dtype

            if np.issubdtype(dtype, np.floating):
                issues.append((p, f'float_range={minv:.6f}:{maxv:.6f}'))
                if FIX:
                    # assume in 0..1 -> scale
                    im2 = np.clip(im, 0.0, 1.0)
                    im2 = (im2 * 255.0).astype(np.uint8)
                    cv2.imwrite(p, im2)
                    print(f'>>>>> Fixed float image (scaled to uint8): {p}')
            else:
                if maxv > 255 or minv < 0:
                    issues.append((p, f'uint_range={minv}:{maxv}'))
        except Exception as e:
            issues.append((p, f'error:{e}'))

    if issues:
        print('----- Image range issues detected (sample):')
        for i, (p, msg) in enumerate(issues[:20]):
            print(f'  {p}: {msg}')
        if not FIX:
            print('\n----- To attempt automatic fixes for float images, run with --fix')
            return 2
    else:
        print('>>>>> All images appear to be uint8 with values in [0,255]')

    return 0

if __name__ == "__main__":
    rc = check_and_fix()
    sys.exit(rc)
