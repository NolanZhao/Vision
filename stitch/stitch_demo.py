import cv2
import numpy as np
import sys
import argparse


class Image_Stitching():
    def __init__(self):
        self.ratio = 0.75
        self.min_match = 10
        self.sift = cv2.xfeatures2d.SIFT_create()
        # self.sift = cv2.xfeatures2d.SURF_create(hessianThreshold=100,
        #                                         nOctaves=1,
        #                                         nOctaveLayers=1,
        #                                         extended=False,
        #                                         upright=True)
        self.smoothing_window_size = 200

    def filter_kp(self, kp, des, w1, w2):
        new_kp = []
        new_des = []
        for i in range(len(kp)):
            if w1 < kp[i].pt[0] < w2:
                new_kp.append(kp[i])
                new_des.append(des[i])
        return new_kp, new_des

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        kp1, des1 = self.filter_kp(kp1, des1, img1.shape[1] * 0.7, img1.shape[1])
        kp2, des2 = self.filter_kp(kp2, des2, 0, img1.shape[1] * 0.3)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        # matcher = cv2.BFMatcher()
        # raw_matches = matcher.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('output/matching.jpg', img3)
        print("Good Match: {}".format(len(good_points)))
        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            # print([i[0] for i in status])
            print('Last Match: ', len([i[0] for i in status if i[0] == 1]))
            return H
        else:
            print("match points less than 10.")
            exit()

    def create_mask(self, img1, img2, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        H = self.registration(img1, img2)

        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))
        panorama2 * mask2
        result = panorama1 + panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result


def main(argv1, argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)

    # img1 = cv2.resize(img1, (480, 480), interpolation=cv2.INTER_AREA)
    # img2 = cv2.resize(img2, (480, 480), interpolation=cv2.INTER_AREA)

    final = Image_Stitching().blending(img1, img2)
    cv2.imwrite('output/panorama.jpg', final)
    print("stitch done.")


if __name__ == '__main__':
    main('images/2imgs/1.jpg', 'images/2imgs/2.jpg')
