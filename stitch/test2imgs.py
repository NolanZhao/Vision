import cv2
import numpy as np


def getH(kp1, kp2, des1, des2):
    # matcher = cv2.BFMatcher()
    matcher = cv2.FlannBasedMatcher()
    raw_matches = matcher.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    good_points = []
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < 0.6 * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])

    if len(good_points) > 12:
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])

        H21, status21 = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        lastmatch21 = len([i[0] for i in status21 if i[0] == 1])

        H12, status12 = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)
        lastmatch12 = len([i[0] for i in status12 if i[0] == 1])

        if lastmatch21 < 10 or lastmatch12 < 10:
            raise Exception("lastmatch less than 10.")

        return H21, H12


def main():
    img1 = cv2.imread("images/2imgs/1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("images/2imgs/2.jpg", cv2.IMREAD_COLOR)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    H21, H12 = getH(kp1, kp2, des1, des2)

    h = img1.shape[0]
    w = img1.shape[1]

    panorama = np.zeros((h, w * 2, 3))
    panorama[0:h, 0:w, :] = img1

    wrap = cv2.warpPerspective(img2, H21, (w * 2, h))
    label = (cv2.cvtColor(wrap[:h, :w * 2, :], cv2.COLOR_BGR2GRAY) > 0)
    for i in range(panorama.shape[2]):
        panorama[:h, :w * 2, i] = wrap[:h, :w * 2, i] * (label > 0) + panorama[:h, :w * 2, i] * (label < 1)

    cv2.imwrite('output/a.jpg', panorama)
    print("stitch done.")


main()
