import os
import torch
import warnings
from model_cnn import MicroDopplerECACNN

warnings.filterwarnings('ignore')

WEIGHTS_FILE = "spectra_eca_best_weights.pt"
ONNX_FILE = "spectra_eca_model.onnx"

def export_model():
    print(f"[SYSTEM] Initializing ONNX Exporter...")

    if not os.path.exists(WEIGHTS_FILE):
        print(f"[!] Error: Cannot find '{WEIGHTS_FILE}'. Did the training finish successfully?")
        return

    device = torch.device("cpu")
    model = MicroDopplerECACNN(num_classes=5)

    print(f"[SYSTEM] Loading trained weights from {WEIGHTS_FILE}...")
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))

    model.eval() 

    dummy_input = torch.randn(1, 1, 128, 128, device=device)
    
    print(f"[SYSTEM] Compiling PyTorch graph to C++ ONNX format...")
    
    # 5. Execute the export
    torch.onnx.export(
        model,                     
        dummy_input,               
        ONNX_FILE,            
        export_params=True,        
        opset_version=11,          
        do_constant_folding=True,  
        input_names=['input_spectrogram'],   
        output_names=['threat_logits'],      
        dynamic_axes={
            'input_spectrogram': {0: 'batch_size'}, 
            'threat_logits': {0: 'batch_size'}
        }
    )
    
    print(f"\n[+] SUCCESS! AI Engine frozen. ONNX model saved as '{ONNX_FILE}'")
    print("[+] The Command Hub is now ready for deployment.")

if __name__ == "__main__":
    export_model()