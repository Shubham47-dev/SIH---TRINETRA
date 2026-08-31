import torch
import onnx
from model_cnn import MicroDopplerECACNN

import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_and_quantize():
    model = MicroDopplerECACNN()
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128)
    onnx_model_path = "spectra_eca_model.onnx"

    print("Exporting PyTorch model to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path, 
        export_params=True,
        opset_version=11,          
        do_constant_folding=True, 
        input_names=['spectrogram'],
        output_names=['classification'],
        dynamic_axes={'spectrogram': {0: 'batch_size'}, 'classification': {0: 'batch_size'}}
    )
    print(f"Standard ONNX model saved to {onnx_model_path}")

    quantized_model_path = "spectra_eca_model_quantized_int8.onnx"
    print("Quantizing ONNX model to INT8...")
    
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized INT8 model saved to {quantized_model_path}")
    print("Model is ready for deployment on Raspberry Pi / Edge Hardware!")

if __name__ == "__main__":
    export_and_quantize()