from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLOv8 Nano model or the best model
    model = YOLO("../model/yolov8n.pt")

    #Train on your custom dataset
    model.train(
        data="data.yaml",
        epochs=500,
        imgsz=640,
        batch=16,
        project="models/",
        name="balloon_v2",
        seed=42,
        device=0,
        workers=2,
        patience=20
    )

    #Evalutation
    # model.val(
    #     data="data.yaml",
    #     device=0,
    #     workers=0,
    #     plots=True,
    # )

    #Inference
    # results = model.predict(
    #     source="videos/WIN_20260908_16_23_32_Pro.mp4", #Change for path that model will inference on
    #     conf=0.5,    # Minimum confidence to display a detection
    #     device=0,     # NVIDIA GPU
    #     save=True,    # Save image with boxes and labels
    #     show=True,
    # )