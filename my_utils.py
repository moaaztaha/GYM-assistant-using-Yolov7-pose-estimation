import matplotlib.pyplot as plt
import torch
import cv2
from typing import Generator, Callable, Tuple
from utils.datasets import letterbox
from torchvision import transforms
import numpy as np
import models.yolo
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from utils.plots import plot_skeleton_kpts

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()


def live_camera(video_processor: Callable = None) -> None:
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if video_processor:
            frame = video_processor(frame)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def calculate_reps(pose_model: models.yolo.Model) -> None:
    feed = cv2.VideoCapture(0)
    if not feed.isOpened():
        print("Cannot open camera")
        exit()

    feed.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stage_left = None
    counter_left = 0
    stage_right = None
    counter_right = 0
    while True:
        success, frame = feed.read()

        if not success:
            print("Can't receive frame, Exiting ..")
            break

        frame = cv2.flip(frame, 1)
        frame, pose_output = process_frame_and_annotate(pose_model, frame, True)
        angle_left = calculate_angle(pose_output, *[6, 8, 10], True, frame)
        angle_right = calculate_angle(pose_output, *[5, 7, 9], True, frame)

        # arm joints
        threshold = 35
        arm_left = calculate_angle(pose_output, *[12, 6, 8], True, frame, 2, threshold)
        arm_right = calculate_angle(pose_output, *[11, 5, 7], True, frame, 2, threshold)

        if arm_left < threshold:
            if angle_left > 150:
                stage_left = 'down'
            if angle_left < 30 and stage_left == 'down':
                stage_left = 'up'
                counter_left += 1
        else:
            stage_left = "skip"
            cv2.putText(frame, "Close your left should!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)

        if arm_right < threshold:
            if angle_right > 150:
                stage_right = 'down'
            if angle_right < 30 and stage_right == 'down':
                stage_right = 'up'
                counter_right += 1
        else:
            stage_right = "skip"
            cv2.putText(frame, "Close your right should!!", (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)

        # drawing stage and counter - left
        cv2.putText(frame, f"Stage: {stage_left}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {str(counter_left)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (225, 225, 225), 3, cv2.LINE_AA)

        # drawing stage and counter - left
        cv2.putText(frame, f"Stage: {stage_right}", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {str(counter_right)}", (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (225, 225, 225), 3, cv2.LINE_AA)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    feed.release()
    cv2.destroyAllWindows()


def plot_image(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.axis('off')
    plt.imshow(image[..., ::-1])  # BGR to RGB
    plt.show()


def load_model(path: str) -> models.yolo.Model:
    weights = torch.load(path, map_location=device)
    model = weights['model'].float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model


POSE_IMAGE_SIZE = 256
STRIDE = 64
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.65


def pose_pre_process_frame(frame: np.ndarray) -> torch.Tensor:
    image = letterbox(frame, POSE_IMAGE_SIZE, stride=STRIDE, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    return image


def clip_coords(boxes: np.ndarray, img_shape: Tuple[int, int]):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def post_process_pose(pose: np.ndarray, image_size: Tuple, scaled_image_size: Tuple) -> np.ndarray:
    height, width = image_size
    scaled_height, scaled_width = scaled_image_size
    vertical_factor = height / scaled_height
    horizontal_factor = width / scaled_width
    result = pose.copy()
    for i in range(17):
        result[i * 3] = horizontal_factor * result[i * 3]
        result[i * 3 + 1] = vertical_factor * result[i * 3 + 1]
    return result


def pose_annotate(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    annotated_frame = image.copy()

    for idx in range(detections.shape[0]):
        pose = detections[idx, 7:].T
        plot_skeleton_kpts(annotated_frame, pose, 3)

    return annotated_frame


def pose_post_process_output(
        model: models.yolo,
        output: torch.tensor,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: Tuple[int, int],
        scaled_image_size: Tuple[int, int]
) -> np.ndarray:
    output = non_max_suppression_kpt(
        prediction=output,
        conf_thres=confidence_threshold,
        iou_thres=iou_threshold,
        nc=model.yaml['nc'],
        nkpt=model.yaml['nkpt'],
        kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

        for idx in range(output.shape[0]):
            output[idx, 7:] = post_process_pose(
                output[idx, 7:],
                image_size=image_size,
                scaled_image_size=scaled_image_size
            )

    return output


def process_frame_and_annotate(model: models.yolo, frame: np.ndarray, return_output: bool = False) -> np.ndarray:
    pose_pre_processed_frame = pose_pre_process_frame(frame=frame.copy())

    image_size = frame.shape[:2]
    scaled_image_size = tuple(pose_pre_processed_frame.size())[2:]

    with torch.no_grad():
        pose_output, _ = model(pose_pre_processed_frame)
        pose_output = pose_post_process_output(
            model=model,
            output=pose_output,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            image_size=image_size,
            scaled_image_size=scaled_image_size
        )

    annotated_frame = pose_annotate(image=frame, detections=pose_output)
    if return_output:
        return annotated_frame, pose_output
    return annotated_frame


def calculate_angle(pose_out: np.ndarray, a: int, b: int, c: int, draw: bool = False,
                    frame: np.ndarray = None, size=3, threshold=None) -> float:
    coord = []
    kpts = pose_out[0, 7:].T
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx_cy = kpts[3 * i], kpts[3 * i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx_cy, conf])

    a = np.array(coord[a][1])
    b = np.array(coord[b][1])
    c = np.array(coord[c][1])

    ba = a - b
    bc = c - b

    cosine_angel = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angel)
    angle = np.degrees(angle)
    if angle > 180:
        angle = 360 - angle

    if draw and frame is not None:
        elbow = int(b[0]), int(b[1])

        if threshold:
            if angle > threshold:
                cv2.putText(frame, str(int(angle)), elbow, cv2.FONT_HERSHEY_SIMPLEX, size*2, (0, 0, 225), 3,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, str(int(angle)), elbow, cv2.FONT_HERSHEY_SIMPLEX, size, (225, 225, 225), 3,
                            cv2.LINE_AA)
        else:
            cv2.putText(frame, str(int(angle)), elbow, cv2.FONT_HERSHEY_SIMPLEX, size, (225, 225, 225), 3, cv2.LINE_AA)

    return angle
