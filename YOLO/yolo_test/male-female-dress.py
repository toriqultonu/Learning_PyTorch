from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(
    data="/home/dexton/Github/Learning_PyTorch/YOLO/yolo_test/data.yaml", epochs=3
)  # train the model
