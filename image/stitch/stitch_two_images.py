import argparse
import os
import sys

from pathlib import Path

import cv2
import numpy as np

from numpy.typing import NDArray


class ImageStitcher:
    def __init__(self,
                 detector: str = 'AKAZE',
                 img_ref: NDArray[np.uint8] | None = None,
                 img_mov: NDArray[np.uint8] | None = None):
        self.detector_name = detector
        self.detector = self._create_detector(detector)
        self.img_ref: NDArray[np.uint8] = img_ref
        self.img_mov: NDArray[np.uint8] = img_mov
        self.keypoints1: list[cv2.KeyPoint] = []
        self.keypoints2: list[cv2.KeyPoint] = []
        self.descriptors1: NDArray[np.uint8] = np.array([], dtype=np.uint8)
        self.descriptors2: NDArray[np.uint8] = np.array([], dtype=np.uint8)
        self.matches: list[tuple[cv2.DMatch], [cv2.DMatch]] = []
        self.good_matches: list[cv2.DMatch] = []

    def _create_detector(self, name: str) -> cv2.Feature2D:
        if name == 'AKAZE':
            return cv2.AKAZE_create()
        elif name == 'ORB':
            return cv2.ORB_create()
        else:
            raise ValueError(f'Unsupported detector: {name}')

    def load_image(self, path: Path | str) -> NDArray[np.uint8]:
        path = Path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Path of reference image is not found: {path}')
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Failed to read image: {path}')
        return img

    def load_image_set(self, path_ref: Path | str, path_mov: Path | str):
        self.img_ref = self.load_image(path_ref)
        self.img_mov = self.load_image(path_mov)

    def compute_and_detect(self,
                           img: NDArray[np.uint8]
                           ) -> tuple[list[cv2.KeyPoint, NDArray[np.uint8]]]:
        keypoints = self.detector.detect(img, None)
        keypoints, descriptors = self.detector.compute(img, keypoints)
        return keypoints, descriptors

    def find_match(self, ratio: float = 0.7) -> list[cv2.DMatch]:
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.matches = flann.knnMatch(self.descriptors1, self.descriptors2, k=2)

        self.good_matches = []
        matchesMask = [[0, 0] for i in range(len(self.matches))]
        for i, m in enumerate(self.matches):
            if len(m) < 2:
                continue
            if m[0].distance < ratio * m[1].distance:
                self.good_matches.append(m[0])
                matchesMask[i] = [1, 0]
        return self.good_matches, matchesMask

    def draw_matches(self, matchesMask: list[tuple[int, int]],
                     output_file: str | Path = 'matching.png'):
        draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=(0, 0, 255),
                matchesMask=matchesMask,
                flags=0
            )
        img_matches = cv2.drawMatchesKnn(
                self.img_ref, self.keypoints1,
                self.img_mov, self.keypoints2,
                self.matches, None, **draw_params
            )
        cv2.imwrite(output_file, img_matches)

    def match(self, draw: bool = False):
        self.keypoints1, self.descriptors1 = self.compute_and_detect(
            self.img_ref)
        self.keypoints2, self.descriptors2 = self.compute_and_detect(
            self.img_mov)
        _, matchesMask = self.find_match()
        if draw:
            self.draw_matches(matchesMask)
        return self.keypoints1, self.keypoints2, self.good_matches

    def homography_matrix(self) -> NDArray[np.float64]:
        pts1 = np.float32(
            [self.keypoints1[m.queryIdx].pt for m in self.good_matches]
            ).reshape(-1, 1, 2)
        pts2 = np.float32(
            [self.keypoints2[m.trainIdx].pt for m in self.good_matches]
            ).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        return homography

    def get_dimension(
            self, H: NDArray) -> tuple[list[int], list[int], NDArray]:
        h1, w1 = self.img_ref.shape[:2]
        h2, w2 = self.img_mov.shape[:2]

        # Calculate size and offset of the merged panorama
        pts = np.array([[0, w2, w2, 0], [0, 0, h2, h2], [1, 1, 1, 1]])
        pts_prime = np.dot(H, pts)
        pts_prime = pts_prime / pts_prime[2]

        # Get width: right edge - left edge
        stitched_w = \
            int(max(pts_prime[0, 1], pts_prime[0, 2], w1)) - \
            int(min(pts_prime[0, 0], pts_prime[0, 3], 0))
        # Get height: bottom edge - top edge
        stitched_h = \
            np.amax(np.int32(pts_prime / pts_prime[2])[1]) - \
            np.amin(np.int32(pts_prime / pts_prime[2])[1], 0)
        stitched = [stitched_w, stitched_h]

        offset = [
            -int(min(pts_prime[0, 0], pts_prime[0, 3], 0)),
            -int(min(pts_prime[1, 0], pts_prime[1, 1], 0))
        ]

        wrap = np.matrix([[1., 0., offset[0]],
                          [0., 1., offset[1]],
                          [0., 0., 1.]])
        T = wrap * H

        return stitched, offset, T

    def stitch(self, output_file: str | Path, draw_matches: bool = False):
        self.match(draw_matches)
        homography = self.homography_matrix()
        size, offset, translation = self.get_dimension(homography)
        panorama = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.warpPerspective(self.img_mov, translation, size, panorama)
        panorama[offset[1]:offset[1] + self.img_ref.shape[0],
                 offset[0]:offset[0] + self.img_ref.shape[1]] = self.img_ref

        cv2.imwrite(output_file, panorama)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Image Stitcher',
        description='Stich the image to the base image and save the result.',
        )
    parser.add_argument('-b', '--base', type=str,
                        help='Path to the base image.')
    parser.add_argument('-s', '--stitched', type=str,
                        help='Path to the stitched image.')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to the result image.')
    parser.add_argument('-d', '--detector', type=str,
                        choices=['AKAZE', 'ORB'],
                        default='AKAZE', help='Detector name.')
    parser.add_argument('-m', '--draw_matches', action='store_true',
                        help='Save keypoints matching in "./matching.png"')
    args = parser.parse_args()

    # Stich images
    img_stitch = ImageStitcher(args.detector)
    img_stitch.load_image_set(args.base, args.stitched)
    img_stitch.stitch(args.output, args.draw_matches)
