"""Execute in an isolated working directory to verify the packaged inference code."""
import json
import sys

import mlflow
import numpy as np
import pandas as pd


def main():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    uri, input_path, expected_path, output_path = sys.argv[1:]
    inputs = pd.read_json(input_path, orient="records")
    with open(expected_path, encoding="utf-8") as stream:
        expected = json.load(stream)
    predictions = mlflow.pyfunc.load_model(uri).predict(inputs)
    if predictions["airline"].tolist() != [row["predicted"] for row in expected["predictions"]]:
        raise AssertionError("Reloaded labels differ from the evaluated checkpoint")
    np.testing.assert_allclose(predictions["confidence"], [row["confidence"] for row in expected["predictions"]],
                               atol=1e-5, rtol=1e-4)
    for item, row in zip(predictions["probabilities_json"], expected["predictions"]):
        vector = json.loads(item)
        np.testing.assert_allclose([vector[name] for name in expected["classes"]], row["probabilities"],
                                   atol=1e-5, rtol=1e-4)
    predictions.to_json(output_path, orient="records")


if __name__ == "__main__":
    main()
