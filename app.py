import uuid

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
from ultralytics import YOLO

MODELS = {
    "YOLO26n  (5.3 MB  · fastest)": "models/yolo26n.pt",
    "YOLO26m  (42.2 MB · balanced)": "models/yolo26m.pt",
    "YOLO26x  (113 MB  · best)": "models/yolo26x.pt",
}
DEFAULT_MODEL = "YOLO26n  (5.3 MB  · fastest)"

_model_cache: dict[str, YOLO] = {}


def get_model(name: str) -> YOLO:
    """Load and cache a YOLO model by display name."""
    if name not in _model_cache:
        _model_cache[name] = YOLO(MODELS[name])
    return _model_cache[name]


def load_model(name: str):
    get_model(name)
    return name, f"✅ **{name.split('(')[0].strip()}** loaded and ready."


# ── helpers ───────────────────────────────────────────────────────────────────

def _annotate(results) -> Image.Image:
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])


# ── inference functions ───────────────────────────────────────────────────────

def predict_image(model_name: str, img: Image.Image, conf: float, iou: float):
    if img is None:
        return None
    results = get_model(model_name).predict(
        source=img, conf=conf, iou=iou, imgsz=640, verbose=False,
    )
    return _annotate(results)


def predict_video(model_name: str, video_path: str, conf: float, iou: float):
    if video_path is None:
        return

    mdl = get_model(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    chunk_size = fps * 2

    def new_writer():
        name = f"det_{uuid.uuid4().hex}.mp4"
        return cv2.VideoWriter(name, fourcc, fps, (w, h)), name

    writer, out_name = new_writer()
    count = 0
    ret, frame = cap.read()

    while ret:
        results = mdl.predict(source=frame, conf=conf, iou=iou, imgsz=640, verbose=False)
        writer.write(results[0].plot())
        count += 1

        if count >= chunk_size:
            writer.release()
            yield out_name
            writer, out_name = new_writer()
            count = 0

        ret, frame = cap.read()

    cap.release()
    writer.release()
    if count > 0:
        yield out_name


def predict_webcam(model_name: str, frame: np.ndarray, conf: float, iou: float):
    if frame is None:
        return None
    results = get_model(model_name).predict(
        source=frame, conf=conf, iou=iou, imgsz=640, verbose=False,
    )
    return _annotate(results)


# ── UI ────────────────────────────────────────────────────────────────────────

def _slider_row():
    conf = gr.Slider(
        0.0, 1.0, value=0.25, step=0.05,
        label="Confidence Threshold",
        info="Minimum score for a detection to be shown. Lower = more boxes, higher = only sure detections.",
    )
    iou = gr.Slider(
        0.0, 1.0, value=0.45, step=0.05,
        label="IoU Threshold",
        info="Controls overlap tolerance between boxes. Lower = merge overlapping boxes more aggressively.",
    )
    return conf, iou


with gr.Blocks(title="YOLO26 Live Object Detection") as app:

    model_state = gr.State(DEFAULT_MODEL)

    gr.HTML("""
        <div style="text-align:center; padding: 16px 0 8px;">
            <h1 style="margin:0; font-size:2rem;">YOLO26 Live Object Detection</h1>
            <p style="color:#6b7280; margin:4px 0 0;">Powered by Ultralytics YOLO26 &amp; Gradio</p>
        </div>
    """)

    # ── Global model selector ─────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Model",
                info="Larger models are more accurate but slower. They auto-download on first use.",
            )
        with gr.Column(scale=1):
            model_btn = gr.Button("Load Model", variant="primary")
    model_status = gr.Markdown(f"✅ **{DEFAULT_MODEL.split('(')[0].strip()}** loaded and ready.")

    model_btn.click(
        fn=load_model,
        inputs=model_dropdown,
        outputs=[model_state, model_status],
    )

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1 · Upload ────────────────────────────────────────────────────
        with gr.Tab("Upload Image / Video"):
            with gr.Tabs():

                # ·· Image ·····················
                with gr.Tab("Image"):
                    with gr.Row():
                        with gr.Column():
                            img_in = gr.Image(type="pil", label="Upload Image")
                            with gr.Row():
                                img_conf, img_iou = _slider_row()
                            img_btn = gr.Button("Detect Objects", variant="primary")
                        with gr.Column():
                            img_out = gr.Image(type="pil", label="Detection Result")

                    img_btn.click(
                        fn=predict_image,
                        inputs=[model_state, img_in, img_conf, img_iou],
                        outputs=img_out,
                    )

                    gr.Examples(
                        examples=[
                            ["samples/test_image.png"],
                            ["samples/test_image_2.jpg"],
                            ["samples/test_image_3.jpg"],
                            ["samples/test_image_4.jpg"],
                            ["samples/test_image_5.jpg"],
                        ],
                        inputs=img_in,
                        label="Sample Images",
                    )

                # ·· Video ·····················
                with gr.Tab("Video"):
                    with gr.Row():
                        with gr.Column():
                            vid_in = gr.Video(label="Upload Video")
                            with gr.Row():
                                vid_conf, vid_iou = _slider_row()
                            vid_btn = gr.Button("Detect Objects", variant="primary")
                        with gr.Column():
                            vid_out = gr.Video(
                                label="Detection Result",
                                streaming=True,
                                autoplay=True,
                            )

                    vid_btn.click(
                        fn=predict_video,
                        inputs=[model_state, vid_in, vid_conf, vid_iou],
                        outputs=vid_out,
                    )

        # ── Tab 2 · Live Webcam ───────────────────────────────────────────────
        with gr.Tab("Live Webcam"):
            gr.Markdown(
                "> Allow browser camera access, then the feed is processed in real time."
            )
            with gr.Row():
                with gr.Column():
                    cam_in = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Webcam Feed",
                    )
                    with gr.Row():
                        cam_conf, cam_iou = _slider_row()
                with gr.Column():
                    cam_out = gr.Image(type="pil", label="Live Detection")

            cam_in.stream(
                fn=predict_webcam,
                inputs=[model_state, cam_in, cam_conf, cam_iou],
                outputs=cam_out,
                time_limit=60,
            )


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
