import numpy as np
import cv2
import matplotlib.pyplot as plt
import sympy


def NMS(boxes, class_ids, confidences, overlapThresh = 0.5):

    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

    # Return empty lists, if no boxes given
    if len(boxes) == 0:
        return [], [], []

    x1 = boxes[:, 0] - (boxes[:, 2] / 2)  # x coordinate of the top-left corner
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  # y coordinate of the top-left corner
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)  # x coordinate of the bottom-right corner
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)  # y coordinate of the bottom-right corner

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices != i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if overlapping greater than our threshold, remove the bounding box
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    # return only the boxes at the remaining indices
    return boxes[indices], class_ids[indices], confidences[indices]


def get_outputs(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    outs = [c for out in outs for c in out if c[4] > 0.1]

    return outs


def draw(bbox, img):

    xc, yc, w, h = bbox
    img = cv2.rectangle(img,
                        (xc - int(w / 2), yc - int(h / 2)),
                        (xc + int(w / 2), yc + int(h / 2)),
                        (0, 255, 0), 20)

    return img


def clean_string(string):
    k = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    getVals = list(filter(lambda x: x in k, string))
    result = "".join(getVals)

    return result


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def dilate(image, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def closing(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)


def padd_image(image, pad_width):
    border_type = cv2.BORDER_CONSTANT

    border_color = [0, 0, 0]

    image_with_padding = cv2.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width,
                                            border_type, value=border_color)
    return image_with_padding

def perspective_transform(image, pts):
    # 520 mm x 114 mm
    dst_pts = np.float32([
        [20, 20],
        [520, 20],
        [520, 114],
        [20, 114]]
    )
    matrix = cv2.getPerspectiveTransform(np.float32(pts), dst_pts)
    return cv2.warpPerspective(image, matrix, (600, 160))


def erode(image, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def convex_hull_image(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    hull = cv2.convexHull(np.vstack(contours))
    hull = hull.reshape((len(hull), 2))

    return hull


def detect_blobs(image):
    output = cv2.connectedComponentsWithStats(image)
    return output


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)

    tl, tr, br, bl = pts

    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))

    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))

    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def get_text_image(image):
    chars_only = np.zeros_like(image)
    total_area = image.shape[0] * image.shape[1]

    blobs = detect_blobs(image)
    (numLabels, labels, stats, centroids) = blobs

    chars_stats = []
    for centroid_idx, centroid_stat in enumerate(stats):
        x, y, w, h, area = centroid_stat

        keepWidth = w < .3 * image.shape[1]
        keepHeight = h > .2 * image.shape[0]
        keepArea = w * h > .02 * total_area  # area < .3 * total_area
        keepOnBorder = x > 0 and y > 0 and x + w < image.shape[1] and y + h < image.shape[0]

        if keepWidth and keepArea and keepOnBorder and keepHeight:
            chars_stats.append((x, y, w, h, centroids[centroid_idx]))
            componentMask = (labels == centroid_idx).astype("uint8") * 255
            chars_only = cv2.bitwise_or(chars_only, componentMask)



    # chars_only_ = dilate(chars_only, iterations=5)
    # chars_only_ = erode(chars_only_, iterations=5)
    # contours, _ = cv2.findContours(chars_only_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     min_area_rect = cv2.minAreaRect(contours[0])
    #
    #     box = np.int0(cv2.boxPoints(min_area_rect))
    #
    #     perspective_image = perspective_transform(chars_only, box)
    #     return perspective_image, chars_only

    if len(chars_stats) > 4:
        chars_stats = np.array(chars_stats)

        min_x_idx, max_x_idx = np.argmin(chars_stats[:, 0]), np.argmax(chars_stats[:, 0])
        pt1 = chars_stats[min_x_idx][:2]
        pt2 = chars_stats[max_x_idx][:2]

        # pt1 = stats[min_x_idx][:2]
        # pt2 = stats[max_x_idx][:2]
        # print(pt1, pt2)
        a = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
        angle = 180 + np.rad2deg(np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0])))
        print(angle)
        height, width = chars_only.shape[:2]
        print()
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        rotated_image = cv2.warpAffine(chars_only, rotation_matrix, (width, height))
        return rotated_image, chars_only

        cv2.circle(chars_only, pt1, 5, (255, 255, 255), -1)
        cv2.circle(chars_only, pt2, 5, (255, 255, 255), -1)

        plt.figure()
        plt.imshow(rotated_image, cmap='gray')
        plt.figure()
        plt.imshow(chars_only, cmap='gray')
        plt.show()
        hull = convex_hull_image(chars_only)

        # for point in hull:
            # cv2.circle(chars_only, point, 4, (255, 255, 255), 2)
        # plt.figure()
        # plt.title("HULL")
        # plt.imshow(cv2.cvtColor(chars_only, cv2.COLOR_BGR2RGB))

        # pt_max_x = np.argwhere(hull[:, 0] == np.argmax(hull[:, 0]))
        # pt_max_y = np.argwhere(hull[:, 1] == np.argmax(hull[:, 1]))
        # pt_min_x = np.argwhere(hull[:, 0] == np.argmin(hull[:, 0]))
        # pt_min_y = np.argwhere(hull[:, 1] == np.argmin(hull[:, 1]))

        # all_center = np.mean(chars_stats[:, -1:])

        # print(all_center)

        # points = list(hull).sort(key=lambda x: np.linalg.norm(np.array(x)-np.array(all_center)))

        # print("POINTS", points)

        pt_max_x = hull[np.argmax(hull[:, 0])]
        pt_max_y = hull[np.argmax(hull[:, 1])]
        pt_min_x = hull[np.argmin(hull[:, 0])]
        pt_min_y = hull[np.argmin(hull[:, 1])]
        print(a)
        if a < -.02:
            pts = [
                pt_min_x,
                pt_min_y,
                pt_max_x,
                pt_max_y
            ]
        elif a > .02:
            pts = [
                pt_min_y,
                pt_max_x,
                pt_max_y,
                pt_min_x
            ]
        else:
            # pt1 = chars_stats[min_x_idx]
            # pt2 = chars_stats[max_x_idx]
            # pt_1 = (pt1[0], pt1[1])
            # pt_2 = (pt2[0] + pt2[2], pt2[1])
            # pt_3 = (pt2[0] + pt2[2], pt2[1] + pt2[3])
            # pt_4 = (pt1[0], pt1[1] + pt1[3])
            # pts = [
            #     pt_1,
            #     pt_2,
            #     pt_3,
            #     pt_4
            # ]
            return chars_only, chars_only

        # for point in pts:
        #     cv2.circle(chars_only, point, 4, (255, 0, 255), 2)
        #     print(point)
        # plt.figure()
        # plt.title("DAWDWA")
        # plt.imshow(cv2.cvtColor(chars_only, cv2.COLOR_BGR2RGB))
        # plt.show()
        perspective_image = perspective_transform(chars_only, pts)

        # plt.figure()
        # plt.title("DAWDWA")
        # plt.imshow(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB))
        # plt.show()


        return perspective_image, chars_only
    else:
        return None, None
