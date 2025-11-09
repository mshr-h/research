import torch
import torchvision
from onnx_to_fx import convert_onnx_to_fx


def main():
    example_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.resnet18(weights=None).eval()

    onnx_model = torch.onnx.export(
        model,
        example_input,
        external_data=False,
    )

    fx_model = convert_onnx_to_fx(onnx_model.model_proto)

    with torch.no_grad():
        original_output = model(example_input)
        fx_output = fx_model(example_input)
        print("First 5 elements of outputs for comparison:")
        print("  Original model:", original_output.flatten()[:5])
        print("  FX model      :", fx_output.flatten()[:5])
        assert torch.allclose(original_output, fx_output, atol=1e-5), (
            "Outputs do not match!"
        )
        print("Outputs match! ✅")


if __name__ == "__main__":
    main()
