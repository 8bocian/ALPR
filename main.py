import argparse
import glob
import os
import shutil
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import utils
from collections import Counter

def majority_voting(texts: [str]):
    max_len = len(max(texts, key=len))

    texts = [text + "_" * (max_len - len(text)) for text in texts]

    cells = np.array([np.array([char for char in text.strip().upper()]) for text in texts])

    output_string = ""

    for position in range(max_len):
        at_position = cells[:, position]

        chars = Counter(at_position).most_common(2)

        winner = chars[0][0] if chars[0][0] != "_" else ""

        output_string += winner

    return output_string


def process_image(img, net):
    found = False
    t = time.time()

    H, W, _ = img.shape

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.merge([img, img, img])

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    net.setInput(blob)

    detections = utils.get_outputs(net)
    described_img = img.copy()
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    bboxes, class_ids, scores = utils.NMS(bboxes, class_ids, scores)

    # reader = easyocr.Reader(['en'])
    for bbox_idx, bbox in enumerate(bboxes):
        b_xc, b_yc, b_w, b_h = bbox
        described_img = cv2.rectangle(described_img,
                                      (int(b_xc - (b_w / 2)), int(b_yc - (b_h / 2))),
                                      (int(b_xc + (b_w / 2)), int(b_yc + (b_h / 2))),
                                      (0, 255, 0),
                                      4)
        padding = 5 #10

        license_plate = img[int(b_yc - (b_h / 2) - padding):int(b_yc + (b_h / 2) + padding),
                        int(b_xc - (b_w / 2) - padding):int(b_xc + (b_w / 2) + (.05 * b_w) + padding),
                        :].copy()

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        license_plate_gray = cv2.fastNlMeansDenoising(license_plate_gray, h=3)

        counts, bins = np.histogram(license_plate_gray, range(257))
        # plot histogram centered on values 0..255


        _, license_plate_thresh = cv2.threshold(license_plate_gray, 96, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        license_plate_thresh = utils.closing(license_plate_thresh)

        # plt.figure()
        # plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        # plt.xlim([-0.5, 255.5])
        # plt.figure()
        # plt.imshow(license_plate_gray, cmap='gray')
        # plt.figure()
        # plt.imshow(license_plate_thresh, cmap='gray')
        # plt.show()
        # plt.figure()
        # plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
        # plt.title("BINARY")

        straight_text_image, chars_only = utils.get_text_image(license_plate_thresh)

        # plt.figure()
        # plt.title("PERSPECTIVE")
        # plt.imshow(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB))

        # plt.figure()
        # plt.title("CHARS")
        # plt.imshow(cv2.cvtColor(chars_only, cv2.COLOR_BGR2RGB))
        # plt.show()

        if straight_text_image is not None:
            config = "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3"

            text = pytesseract.pytesseract.image_to_string(straight_text_image, lang="eng", config=config)
            clear_text = utils.clean_string(text)

            cv2.putText(described_img,
                        clear_text,
                        (int(b_xc - (b_w / 2)), int(b_yc - (b_h / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (0, 255, 0),
                        15)
            found = True

        # plt.figure()
        # plt.title(scores[bbox_idx])
        # plt.imshow(cv2.cvtColor(described_img, cv2.COLOR_BGR2RGB))
        # plt.show()
        # plt.show()

    time_ = time.time() - t

    # plt.figure()
    # plt.imshow(cv2.cvtColor(described_img, cv2.COLOR_BGR2RGB))
    # plt.show()

    if found:
        return time_, img, described_img, text.strip(), license_plate_thresh, chars_only, straight_text_image, license_plate_gray
    else:
        return time_, img, described_img, None, None, None, None, None


def process_batch(path, net, verbose=False, save_to="video.mp4"):
    if path.endswith(".mp4"):
        shutil.rmtree("errors")
        os.mkdir("errors")
        texts = []
        cap = cv2.VideoCapture(path)

        if (cap.isOpened() == False):
            print("Error reading video file")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        size = (frame_width, int(frame_height))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        result = cv2.VideoWriter(save_to, fourcc=fourcc, frameSize=size, fps=fps)

        frame_idx = 0
        is_next_frame = True
        while is_next_frame:
            if cap.grab():
                flag, frame = cap.retrieve()
                if not flag:
                    continue
                else:
                    time, image, described_img, text, license_plate_thresh, chars_only, perspective_image, gray_scale_image = process_image(frame, net)
                    if text != None and text != "PGN428CG":
                        l = len(os.listdir("errors"))+1
                        directory = f"errors/error{l}/"
                        os.mkdir(directory)
                        filename = directory + "license_plate_thresh.png"
                        cv2.imwrite(filename, license_plate_thresh)
                        filename = directory + "chars_only.png"
                        cv2.imwrite(filename, chars_only)
                        filename = directory + "gray_scale_image.png"
                        cv2.imwrite(filename, gray_scale_image)
                        cv2.putText(perspective_image,
                                    text,
                                    (0, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    1)
                        filename = directory + "perspective_image.png"
                        cv2.imwrite(filename, perspective_image)
                    if text is None:
                        print(f"Plates not found on frame {frame_idx}")
                    else:
                        texts.append(text)
                        major_text = majority_voting(texts)
                        print(f"Result: {major_text}")

                        # image = cv2.cvtColor(image, cv2)
                        concatenated_images = cv2.vconcat(perspective_image, license_plate_thresh, )
                        concatenated_images = cv2.cvtColor(concatenated_images, cv2.COLOR_GRAY2BGR)
                        if verbose == 1:
                            ...
                        elif verbose == 2:
                            plt.figure()
                            plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
                            plt.figure()
                            plt.imshow(cv2.cvtColor(chars_only, cv2.COLOR_BGR2RGB))
                            plt.figure()
                            plt.imshow(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB))
                            plt.show()
                    # cv2.imshow('video', described_img)
                    result.write(described_img)

                    print(f"{1 / time:.2f} FPS, Time: {time:.2f}s")
            else:
                print("SAVING")
                cap.release()
                result.release()

                cv2.destroyAllWindows()
                return
            if verbose:
                if cv2.waitKey(10) == 27:
                    break
            frame_idx += 1
            print(f"Progress: {(frame_idx / total_frames) * 100:.2f}%")


    else:
        img = cv2.imread(path)

        time, image, described_img, text, license_plate_thresh, chars_only, perspective_image  = process_image(img, net)
        # print(text)
        if verbose:
            plt.figure()
            plt.title(text)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            plt.figure()
            plt.imshow(license_plate_thresh)

            plt.figure()
            plt.title("chars_only")
            plt.imshow(chars_only)

            plt.figure()
            plt.imshow(perspective_image)

            plt.show()
        if text is None:
            print(f"Plates not found on image")


if __name__ == "__main__":
    model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
    class_names_path = os.path.join('.', 'model', 'class.names')

    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    parser = argparse.ArgumentParser(description="A script with --path argument")
    parser.add_argument("--path", help="Specify the video path", required=True)
    parser.add_argument("--output_path", help="Specify the video output path", required=True)

    args = parser.parse_args()
    path = args.path
    out_path = args.output_path

    process_batch(path, net, verbose=False, save_to=out_path)

