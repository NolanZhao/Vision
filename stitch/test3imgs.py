import cv2
import numpy as np


def getH(kp1, kp2, des1, des2):
    # matcher = cv2.BFMatcher()
    matcher = cv2.FlannBasedMatcher()
    raw_matches = matcher.knnMatch(np.asarray(des1, np.float32),
                                   np.asarray(des2, np.float32),
                                   k=2)

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
    img1 = cv2.imread("images/3imgs/1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("images/3imgs/2.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("images/3imgs/3.jpg", cv2.IMREAD_COLOR)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=999999,
                                       nOctaveLayers=6,
                                       contrastThreshold=0.004,
                                       edgeThreshold=3)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)

    H21, H12 = getH(kp1, kp2, des1, des2)
    H32, H23 = getH(kp2, kp3, des2, des3)

    h = img1.shape[0]
    w = img1.shape[1]

    panorama = np.zeros((h, w * 3, 3))
    panorama[0:h, w:2 * w, :] = img2

    t = np.array([[1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    H32 = np.dot(H32, t)
    H12 = np.dot(H12, t)

    wrap1 = cv2.warpPerspective(img1, H12, (w * 3, h))
    label1 = (cv2.cvtColor(wrap1[:h, :w * 3, :], cv2.COLOR_BGR2GRAY) > 0)
    for i in range(panorama.shape[2]):
        panorama[:h, :w * 3, i] = wrap1[:h, :w * 3, i] * (
            label1 > 0) + panorama[:h, :w * 3, i] * (label1 < 1)

    wrap3 = cv2.warpPerspective(img3, H32, (w * 3, h))
    label3 = (cv2.cvtColor(wrap3[:h, :w * 3, :], cv2.COLOR_BGR2GRAY) > 0)
    for i in range(panorama.shape[2]):
        panorama[:h, :w * 3, i] = wrap3[:h, :w * 3, i] * (
            label3 > 0) + panorama[:h, :w * 3, i] * (label3 < 1)

    cv2.imwrite('output/a.jpg', panorama)


main()
