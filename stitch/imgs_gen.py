import copy
import cv2


def devide_2x3():
    source_img = cv2.imread("images/source/s1.jpg", cv2.IMREAD_COLOR)
    h, w, _ = source_img.shape

    img0 = copy.deepcopy(source_img)
    crop_img0 = img0[0:h // 2 + h // 5, 0:w // 3 + w // 5]
    cv2.imwrite("0.jpg", crop_img0)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[h // 2 - h // 5:h, 0:w // 3 + w // 5]
    cv2.imwrite("1.jpg", crop_img1)

    img2 = copy.deepcopy(source_img)
    crop_img2 = img2[0:h // 2 + h // 5, w // 3 - w // 10:2 * w // 3 + w // 10]
    cv2.imwrite("2.jpg", crop_img2)

    img3 = copy.deepcopy(source_img)
    crop_img3 = img3[h // 2 - h // 5:h, w // 3 - w // 10:2 * w // 3 + w // 10]
    cv2.imwrite("3.jpg", crop_img3)

    img4 = copy.deepcopy(source_img)
    crop_img4 = img4[0:h // 2 + h // 5, 2 * w // 3 - w // 5:w]
    cv2.imwrite("4.jpg", crop_img4)

    img5 = copy.deepcopy(source_img)
    crop_img5 = img5[h // 2 - h // 5:h, 2 * w // 3 - w // 5:w]
    cv2.imwrite("5.jpg", crop_img5)


def devide_2():
    source_img = cv2.imread("images/source/s1.jpg", cv2.IMREAD_COLOR)
    h, w, _ = source_img.shape

    img0 = copy.deepcopy(source_img)
    crop_img0 = img0[0:h, 0:3*w//5]
    cv2.imwrite("1.jpg", crop_img0)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, 2*w//5:w]
    cv2.imwrite("2.jpg", crop_img1)


def devide_3():
    source_img = cv2.imread("images/source/s1.jpg", cv2.IMREAD_COLOR)
    h, w, _ = source_img.shape
    delta = 60

    img0 = copy.deepcopy(source_img)
    crop_img0 = img0[0:h, 0:int((1/3) * w + delta)]
    cv2.imwrite("1.jpg", crop_img0)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, int((1/3) * w - delta / 2):int((2/3) * w + delta / 2)]
    cv2.imwrite("2.jpg", crop_img1)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, int((2/3) * w - delta / 2):w]
    cv2.imwrite("3.jpg", crop_img1)


def devide_5():
    source_img = cv2.imread("images/source/s1.jpg", cv2.IMREAD_COLOR)
    h, w, _ = source_img.shape
    delta = 40

    img0 = copy.deepcopy(source_img)
    crop_img0 = img0[0:h, 0:int((1/5) * w + delta)]
    cv2.imwrite("1.jpg", crop_img0)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, int((1/5) * w - delta / 2):int((2/5) * w + delta / 2)]
    cv2.imwrite("2.jpg", crop_img1)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, int((2/5) * w - delta / 2):int((3/5) * w + delta / 2)]
    cv2.imwrite("3.jpg", crop_img1)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, int((3/5) * w - delta / 2):int((4/5) * w + delta / 2)]
    cv2.imwrite("4.jpg", crop_img1)

    img1 = copy.deepcopy(source_img)
    crop_img1 = img1[0:h, int((4/5) * w - delta):w]
    cv2.imwrite("5.jpg", crop_img1)




devide_5()