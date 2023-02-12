from my_utils import load_model, calculate_reps

POSE_MODEL_WEIGHTS_PATH = "yolov7-w6-pose.pt"
pose_model = load_model(POSE_MODEL_WEIGHTS_PATH)

print(f"Loaded model {POSE_MODEL_WEIGHTS_PATH}")

calculate_reps(pose_model)
