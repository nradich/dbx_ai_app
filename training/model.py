"""MLflow Models-from-Code entry point; input images must focus on the tail."""
import json

import mlflow
import pandas as pd
import torch

from vision import build_model, decode_image, evaluation_transform


class AirlineTailModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        checkpoint = torch.load(context.artifacts["checkpoint"], map_location="cpu", weights_only=True)
        self.classes = checkpoint["classes"]
        self.model = build_model(len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = evaluation_transform()

    def predict(self, context, model_input, params=None):
        outputs = []
        values = model_input["image_base64"].tolist()
        with torch.inference_mode():
            for offset in range(0, len(values), 16):
                batch = torch.stack([
                    self.transform(decode_image(value)) for value in values[offset:offset + 16]
                ])
                for probabilities in self.model(batch).softmax(-1):
                    index = probabilities.argmax().item()
                    outputs.append({
                        "airline": self.classes[index],
                        "confidence": float(probabilities[index]),
                        "probabilities_json": json.dumps(dict(zip(self.classes, probabilities.tolist()))),
                    })
        return pd.DataFrame(outputs, columns=["airline", "confidence", "probabilities_json"])


mlflow.models.set_model(AirlineTailModel())
