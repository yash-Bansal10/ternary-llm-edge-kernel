import os
import struct
import numpy as np
import torch
try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

def load_ternary_tensor(file_path=None, tensor_name=None, shape=(512, 16384)):
    """
    Loads a tensor from a safetensors file, or falls back to a 
    synthetic BitNet distribution matrix to preserve local disk storage.
    """
    if file_path and os.path.exists(file_path) and SAFETENSORS_AVAILABLE:
        print(f"[*] Loading real tensor from {file_path}...")
        state_dict = load_file(file_path)
        if tensor_name and tensor_name in state_dict:
            tensor = state_dict[tensor_name].float()
        else:
            weight_keys = [k for k in state_dict.keys() if "weight" in k]
            tensor = state_dict[weight_keys[0]].float()
        
        # Clamp to {-1, 0, 1} and ensure it is cast to int8
        tensor = torch.round(torch.clamp(tensor, -1.0, 1.0))
        return tensor.cpu().numpy().astype(np.int8)
    else:
        print(f"[!] Simulating a real 1.58-bit BitNet Layer ({shape[0]}x{shape[1]})...")
        print(f"    (No valid safetensors file provided. Generating in RAM to save disk space).")
        rng = np.random.default_rng(seed=42)
        raw_data = rng.choice([-1, 0, 1], size=shape, p=[0.33, 0.34, 0.33])
        return raw_data.astype(np.int8)

def pack_ternary_weights(weights_matrix):
    """
    Packs an (M, N) matrix into dual-plane uint32 bitmasks using high-speed NumPy vectorization.
    """
    rows, cols = weights_matrix.shape
    assert cols % 32 == 0, "Matrix columns must be a multiple of 32 for word alignment!"
    words_per_row = cols // 32
    
    pos_packed = np.zeros((rows, words_per_row), dtype=np.uint32)
    neg_packed = np.zeros((rows, words_per_row), dtype=np.uint32)
    
    print(f"[*] Packing {rows}x{cols} weights into {rows}x{words_per_row} bitmask words (Vectorized Core)...")
    
    # Vectorized loop: evaluates entire columns simultaneously instead of individual scalars
    for bit_idx in range(32):
        col_indices = np.arange(bit_idx, cols, 32)
        weight_slice = weights_matrix[:, col_indices]
        
        pos_packed |= ((weight_slice == 1).astype(np.uint32) << bit_idx)
        neg_packed |= ((weight_slice == -1).astype(np.uint32) << bit_idx)
            
    return pos_packed, neg_packed

def verify_and_unpack(pos_packed, neg_packed, original_matrix):
    """
    Vectorized unpacking to confirm 100% data integrity instantly.
    """
    print("[*] Initiating Round-Trip Integrity Verification (Vectorized)...")
    rows, words_per_row = pos_packed.shape
    cols = words_per_row * 32
    unpacked_matrix = np.zeros((rows, cols), dtype=np.int8)
    
    for bit_idx in range(32):
        col_indices = np.arange(bit_idx, cols, 32)
        
        p_bit = ((pos_packed >> bit_idx) & 1).astype(np.int8)
        n_bit = ((neg_packed >> bit_idx) & 1).astype(np.int8)
        
        unpacked_matrix[:, col_indices] = p_bit - n_bit
        
    mismatches = np.sum(unpacked_matrix != original_matrix)
    if mismatches == 0:
        print(" [PASS] Verification Complete. Mathematical parity is 100% verified.")
        return True
    else:
        print(f" [FAIL] WARNING! Found {mismatches} mismatching elements!")
        return False

def export_to_binary(pos_packed, neg_packed, output_filename="packed_weights.bin"):
    """Saves packed arrays into a raw binary stream for C++ consumption."""
    rows, words_per_row = pos_packed.shape
    print(f"[*] Exporting compressed layout to '{output_filename}'...")
    with open(output_filename, "wb") as f:
        f.write(struct.pack("II", rows, words_per_row))
        f.write(pos_packed.tobytes())
        f.write(neg_packed.tobytes())
    print(f" [SUCCESS] Binary footprint: {os.path.getsize(output_filename) / 1024:.2f} KB")

if __name__ == "__main__":
    # Point this to your local Llama-3/BitNet safetensors file
    SAFETENSORS_PATH = r"model.safetensors" # This will be the path to the downloaded HuggingFace model
    TENSOR_NAME = None       
    OUTPUT_BIN_PATH = "packed_weights.bin"
    MATRIX_SHAPE = (512, 16384) 
    
    weights = load_ternary_tensor(SAFETENSORS_PATH, TENSOR_NAME, shape=MATRIX_SHAPE)
    pos_packed, neg_packed = pack_ternary_weights(weights)
    if verify_and_unpack(pos_packed, neg_packed, weights):
        export_to_binary(pos_packed, neg_packed, OUTPUT_BIN_PATH)
