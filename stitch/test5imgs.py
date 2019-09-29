import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
import copy


class Point(object):
    def __init__(self, x=0.0, y=0.0, z=1.0):
        self.x = x
        self.y = y
        self.z = z

    def calculatenewpoint(self, homo):
        point_old = np.array([self.x, self.y, self.z]).reshape(3, 1)
        point_new = np.dot(homo, point_old)
        point_new /= point_new[2, 0]
        self.x = point_new[0, 0]
        self.y = point_new[1, 0]
        point_new = np.dot(homo, point_old)
        self.z = point_new[2, 0]


class Corner(object):
    def __init__(self):
        self.ltop = Point()
        self.lbottom = Point()
        self.rtop = Point()
        self.rbottom = Point()

    def calculatefromimage(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        self.ltop.x = 0.0
        self.ltop.y = 0.0
        self.lbottom.x = 0.0
        self.lbottom.y = float(rows)
        self.rtop.x = float(cols)
        self.rtop.y = 0.0
        self.rbottom.x = float(cols)
        self.rbottom.y = float(rows)

    def calculatefromhomo(self, homo):
        self.ltop.calculatenewpoint(homo)
        self.lbottom.calculatenewpoint(homo)
        self.rtop.calculatenewpoint(homo)
        self.rbottom.calculatenewpoint(homo)

    def getoutsize(self):
        lx = min(self.ltop.x, self.lbottom.x)
        rx = max(self.rtop.x, self.rbottom.x)
        uy = min(self.ltop.y, self.rtop.y)
        dy = max(self.lbottom.y, self.rbottom.y)
        return lx, rx, uy, dy


def calculatecorners(imgs, homos):
    result = list()
    for index, img in enumerate(imgs):
        c = Corner()
        c.calculatefromimage(img)
        c.calculatefromhomo(homos[index])
        result.append(c)
    return result


def get_homography(img1, img2, method=0):
    """
    method: 0 findHomography
            1 estimateAffine2D
    """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=999999, nOctaveLayers=6, contrastThreshold=0.004, edgeThreshold=3)

    img1_ = copy.deepcopy(img1)
    img2_ = copy.deepcopy(img2)

    kp1, des1 = sift.detectAndCompute(img1_, None)
    kp2, des2 = sift.detectAndCompute(img2_, None)

    matcher = cv2.FlannBasedMatcher()
    raw_matches = matcher.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
    good_matches = [m1 for m1, m2 in raw_matches if m1.distance < 0.6 * m2.distance]

    image1_kp = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    image2_kp = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    if method == 0:
        H21, status21 = cv2.findHomography(image2_kp,
                                           image1_kp,
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=16,
                                           confidence=0.8)
        lastmatch21 = len([i[0] for i in status21 if i[0] == 1])
        H12, status12 = cv2.findHomography(image1_kp,
                                           image2_kp,
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=16,
                                           confidence=0.8)
        lastmatch12 = len([i[0] for i in status12 if i[0] == 1])
    else:
        H21_, status21 = cv2.estimateAffine2D(image2_kp,
                                              image1_kp,
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=16,
                                              confidence=0.8)
        lastmatch21 = len([i[0] for i in status21 if i[0] == 1])
        H12_, status12 = cv2.estimateAffine2D(image1_kp,
                                              image2_kp,
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=16,
                                              confidence=0.8)
        lastmatch12 = len([i[0] for i in status12 if i[0] == 1])

        H21 = np.zeros((3, 3), dtype=float)
        H21[2, 2] = 1.0
        H21[:H21_.shape[0], :H21_.shape[1]] = H21_

        H12 = np.zeros((3, 3), dtype=float)
        H12[2, 2] = 1.0
        H12[:H12_.shape[0], :H12_.shape[1]] = H12

    if lastmatch21 < 10 or lastmatch12 < 10:
        print("lastmatch21: ", lastmatch21)
        print("lastmatch12: ", lastmatch12)
        raise Exception("lastmatch less than 10.")

    return H21, H12


def wrapimgs(imgs, homos, corners):
    min_up = 0.0
    for i, corner in enumerate(corners):
        up = min(corners[i].ltop.y, corners[i].rtop.y)
        min_up = min(min_up, up)
    tem = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -min_up], [0.0, 0.0, 1.0]])
    wraps = list()
    for index, img in enumerate(imgs):
        lx, rx, uy, dy = corners[index].getoutsize()
        cols = int(rx - lx * (lx < 0))
        rows = int(dy) - int(min_up)
        wrap = cv2.warpPerspective(imgs[index], np.dot(tem, homos[index]), (cols, rows))
        # cv2.imwrite('wrapimgs_{}.jpg'.format(index), wrap)
        wraps.append(wrap)
    return wraps


def putimages(wraps):
    h = max(*[wrap.shape[0] for wrap in wraps])
    w = max(*[wrap.shape[1] for wrap in wraps])
    stitch_im = np.zeros((h, w, 3), dtype='uint8')
    wraps_ = wraps[::-1]
    for index, wrap in enumerate(wraps_):
        h = wrap.shape[0]
        w = wrap.shape[1]
        label = (cv2.cvtColor(wrap[:h, :w, :], cv2.COLOR_BGR2GRAY) > 0)
        for i in range(stitch_im.shape[2]):
            stitch_im[:h, :w, i] = wrap[:h, :w, i] * \
                (label > 0) + stitch_im[:h, :w, i] * (label < 1)
    result = stitch_im
    return result


def main():
    img1 = cv2.imread("images/5imgs/1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("images/5imgs/2.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("images/5imgs/3.jpg", cv2.IMREAD_COLOR)
    img4 = cv2.imread("images/5imgs/4.jpg", cv2.IMREAD_COLOR)
    img5 = cv2.imread("images/5imgs/5.jpg", cv2.IMREAD_COLOR)

    imgs = [img1, img2, img3, img4, img5]

    H21, H12 = get_homography(img1, img2)
    H32, H23 = get_homography(img2, img3)
    H43, H34 = get_homography(img3, img4)
    H54, H45 = get_homography(img4, img5)

    H13 = np.dot(H23, H12)
    H53 = np.dot(H43, H54)

    h = img1.shape[0]
    w = img1.shape[1]

    panorama = np.zeros((h, w * 5, 3))
    panorama[0:h, 2 * w:3 * w, :] = img3

    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    a = np.dot(H13, np.array([0.0, 0.0, 1.0]).reshape(3, 1))
    a = a[0, 0] / a[2, 0]
    b = np.dot(H13, np.array([0.0, h, 1.0]).reshape(3, 1))
    b = b[0, 0] / b[2, 0]
    c = abs(min(a, b))
    Hpivot = np.array([[1.0, 0.0, float(c)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    homos_ = [H13, H23, H, H43, H53]
    homos = [np.dot(Hpivot, X) for X in homos_]

    corners = calculatecorners(imgs, homos)
    print("corners done.")

    wraps = wrapimgs(imgs, homos, corners)
    print("wraps done.")

    stitch_im = putimages(wraps)
    cv2.imwrite('output/a.jpg', stitch_im)
    print("stitch done.")


main()
