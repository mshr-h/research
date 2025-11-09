# SPDX-License-Identifier: Apache-2.0
import torch
from onnx_to_fx import convert_onnx_to_fx


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(4, 3)
        self.fc2 = torch.nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def main():
    example_input = torch.randn(1, 4)
    model = MLP()
    print(model)

    onnx_model = torch.onnx.export(
        model,
        example_input,
        external_data=False,
    )

    fx_model = convert_onnx_to_fx(onnx_model.model_proto)
    print(fx_model)

    with torch.no_grad():
        original_output = model(example_input)
        fx_output = fx_model(example_input)
        print("Original model output:", original_output)
        print("FX model output:", fx_output)
        assert torch.allclose(original_output, fx_output, atol=1e-5), (
            "Outputs do not match!"
        )
        print("Outputs match! ✅")


if __name__ == "__main__":
    main()
