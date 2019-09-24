import numpy as np
import cv2


def get_matches(img1, img2, mode="LR"):
    if mode == "LR":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=999999, nOctaveLayers=6, contrastThreshold=0.004, edgeThreshold=3)
    else:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=999999, nOctaveLayers=9, contrastThreshold=0.0001, edgeThreshold=2)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.FlannBasedMatcher()
    raw_matches = matcher.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
    good_matches = [m1 for m1, m2 in raw_matches if m1.distance < 0.6 * m2.distance]

    image1_kp = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    image2_kp = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    res1, res2 = [], []
    H21, status21 = cv2.findHomography(image2_kp,
                                       image1_kp,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=16,
                                       confidence=0.8)
    for i in range(len(status21)):
        if status21[i][0] == 1:
            res1.append(image1_kp[i])
            # res2.append(image2_kp[i])

    H12, status12 = cv2.findHomography(image1_kp,
                                       image2_kp,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=16,
                                       confidence=0.8)
    for i in range(len(status12)):
        if status21[i][0] == 1:
            # res1.append(image1_kp[i])
            res2.append(image2_kp[i])

    return res1, res2


def coordinate_transform(L, delta, shape, imgs, position1, position2):
    new_L = []
    row1, col1 = position1
    ind1 = col1 * shape[0] + row1
    img1 = imgs[ind1]

    row2, col2 = position2
    ind2 = col2 * shape[0] + row2
    img2 = imgs[ind2]
    for (pt1, pt2) in L:
        x1, y1 = pt1[0], pt1[1]
        x1 += (img1.shape[1] + delta) * col1
        y1 += (img1.shape[0] + delta) * row1

        x2, y2 = pt2[0], pt2[1]
        x2 += (img2.shape[1] + delta) * col2
        y2 += (img2.shape[0] + delta) * row2

        new_L.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return new_L


def draw_match(shape, imgs):
    h, w, _ = imgs[0].shape
    delta = 200
    panorama = np.zeros(((h + delta) * shape[0], (w + delta) * shape[1], 3))

    matches = {}

    for i in range(shape[0]):
        for j in range(shape[1] - 1):
            ind1, ind2 = shape[0] * j + i, shape[0] * (j + 1) + i
            print("get match: ", ind1, ind2)
            matches[f"{ind1}{ind2}"] = get_matches(imgs[ind1], imgs[ind2])

    for i in range(shape[1]):
        c_imgs = imgs[i * shape[0]:(i + 1) * shape[0]]
        img = imgs[i * shape[0]]
        panorama[0:img.shape[0], (img.shape[1] + delta) * i:(img.shape[1] + delta) * i + img.shape[1], :] = img

        for j in range(1, len(c_imgs)):
            ind1 = i * shape[0] + j - 1
            ind2 = i * shape[0] + j
            print("get match: ", ind1, ind2)
            matches[f"{ind1}{ind2}"] = get_matches(imgs[ind1], imgs[ind2], mode="UD")

            img_ = imgs[ind2]
            panorama[(img_.shape[0] + delta) * j:(img_.shape[0] + delta) * j + img_.shape[0], (img_.shape[1] + delta) *
                     i:(img_.shape[1] + delta) * i + img_.shape[1], :] = img_

    # draw lines
    for i in range(shape[0]):
        for j in range(shape[1] - 1):
            ind1, ind2 = shape[0] * j + i, shape[0] * (j + 1) + i
            a, b = matches[f"{ind1}{ind2}"]
            data = coordinate_transform(list(zip(a, b)), delta, shape, imgs, (i, j), (i, j + 1))
            for pts in data:
                cv2.line(panorama, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2)

    for i in range(shape[1]):
        c_imgs = imgs[i * shape[0]:(i + 1) * shape[0]]
        for j in range(1, len(c_imgs)):
            ind1, ind2 = i * shape[0] + j - 1, i * shape[0] + j
            a, b = matches[f"{ind1}{ind2}"]
            data = coordinate_transform(list(zip(a, b)), delta, shape, imgs, (j - 1, i), (j, i))
            for pts in data:
                cv2.line(panorama, tuple(pts[0]), tuple(pts[1]), (255, 255, 0), 2)

    cv2.imwrite('output/matches.jpg', panorama)

    return None, None


if __name__ == "__main__":
    img1 = cv2.imread("images/5imgs/1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("images/5imgs/2.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("images/5imgs/3.jpg", cv2.IMREAD_COLOR)
    img4 = cv2.imread("images/5imgs/4.jpg", cv2.IMREAD_COLOR)
    img5 = cv2.imread("images/5imgs/5.jpg", cv2.IMREAD_COLOR)

    imgs = [img1, img2, img3, img4, img5]
    draw_match((1, 5), imgs)