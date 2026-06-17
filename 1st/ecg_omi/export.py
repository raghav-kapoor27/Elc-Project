from __future__ import annotations

from pathlib import Path

import torch

from ecg_omi.config import DEFAULT_WINDOW_SAMPLES
from ecg_omi.models.foundation import ECGFoundationModel


class ExportableFoundation(torch.nn.Module):
    def __init__(self, model: ECGFoundationModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["embedding"], out["signal_quality"], out["noise_score"]


def load_model(checkpoint: str | Path, device: str = "cpu") -> ECGFoundationModel:
    model = ECGFoundationModel().to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("state_dict", state)
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def export_all(checkpoint: str | Path, out_dir: str | Path = "exports") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExportableFoundation(load_model(checkpoint))
    dummy = torch.randn(1, 1, DEFAULT_WINDOW_SAMPLES)
    torch.onnx.export(
        model,
        dummy,
        out / "foundation.onnx",
        input_names=["ecg"],
        output_names=["embedding", "signal_quality", "noise_score"],
        dynamic_axes={"ecg": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(out / "foundation.ts"))
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(out / "foundation.onnx")
        tf_rep = prepare(onnx_model)
        saved_model_dir = out / "foundation_saved_model"
        tf_rep.export_graph(str(saved_model_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tflite_model = converter.convert()
        (out / "foundation.tflite").write_bytes(tflite_model)
    except Exception as exc:
        (out / "TFLITE_EXPORT_NOTE.txt").write_text(
            "TFLite export requires compatible onnx-tf and tensorflow versions.\n"
            f"ONNX and TorchScript exports succeeded. Error: {exc}\n",
            encoding="utf-8",
        )
