import struct
import random
import os

# ==========================================
# [Global Configuration] Single source of truth
# Modify here, generation and verification will both take effect
# ==========================================
class ADMConfig:
    BLOCK_SIZE = 512    # block_size in the C++ code
    THRESHOLD  = 1000   # Maximum allowed range (max - min) in ADM
    
    # Data type configuration
    IS_U16     = True   # True=uint16, False=uint32

# ==========================================
# 1. Generator
# ==========================================
def generate_dataset(filename, total_elements, config=ADMConfig):
    """
    Generate a dataset with arbitrary length.
    Logic: strictly split by config.BLOCK_SIZE, for each block choose a
    separate base value, and ensure Max-Min <= config.THRESHOLD inside
    each block (including the tail block that is shorter than BLOCK_SIZE).
    """
    # Decide value range and struct format according to type
    if config.IS_U16:
        MAX_VAL = 65535
        fmt_char = 'H'
    else:
        MAX_VAL = 4294967295
        fmt_char = 'I'
        
    # print(f"[Generator] Creating '{filename}'...")
    # print(f"  - Total Elements: {total_elements}")
    # print(f"  - Block Size:     {config.BLOCK_SIZE}")
    # print(f"  - Threshold:      {config.THRESHOLD}")

    with open(filename, 'wb') as f:
        # Core logic: step must equal BLOCK_SIZE to match the C++ reading logic
        for i in range(0, total_elements, config.BLOCK_SIZE):
            
            # Compute current block size (handle tail block smaller than 512)
            current_chunk_len = min(config.BLOCK_SIZE, total_elements - i)
            
            # For the current block, independently choose a base value
            # to keep the range of this block under control
            current_base = random.randint(0, MAX_VAL - config.THRESHOLD)
            
            chunk_data = []
            for _ in range(current_chunk_len):
                # Data strictly falls into [base, base + threshold]
                val = random.randint(current_base, current_base + config.THRESHOLD)
                chunk_data.append(val)
            
            # Write as binary
            f.write(struct.pack(f'{current_chunk_len}{fmt_char}', *chunk_data))

    # print(f"[Generator] Done.\n")

# ==========================================
# 2. Verifier
# ==========================================
def verify_dataset(filename, config=ADMConfig):
    print(f"[Verifier] Checking '{filename}'...")
    
    if not os.path.exists(filename):
        print("File not found!")
        return

    with open(filename, 'rb') as f:
        content = f.read()

    # Parse parameters
    ele_size = 2 if config.IS_U16 else 4
    fmt_char = 'H' if config.IS_U16 else 'I'
    
    total_bytes = len(content)
    if total_bytes % ele_size != 0:
        print("Error: File size is not aligned with element size.")
        return

    num_elements = total_bytes // ele_size
    data = struct.unpack(f'{num_elements}{fmt_char}', content)
    
    global_max_diff = 0
    fail_count = 0
    
    # Core logic: simulate C++ reading, step = config.BLOCK_SIZE
    for i in range(0, num_elements, config.BLOCK_SIZE):
        # Slice (Python slicing handles out-of-range automatically,
        # equivalent to using min for the end index)
        chunk = data[i : i + config.BLOCK_SIZE]
        
        bmin = min(chunk)
        bmax = max(chunk)
        diff = bmax - bmin
        
        if diff > global_max_diff:
            global_max_diff = diff
            
        if diff > config.THRESHOLD:
            print(f"  \033[91m[FAIL]\033[0m Block Idx {i//config.BLOCK_SIZE} (Start {i}, Len {len(chunk)})")
            print(f"         Min: {bmin}, Max: {bmax}, Diff: {diff} (Threshold: {config.THRESHOLD})")
            fail_count += 1
            if fail_count > 5:
                print("  ... Too many failures, aborting check ...")
                break
    
    print(f"[Verifier] Global Max Diff found: {global_max_diff}")
    
    if fail_count == 0:
        print(f"\033[92m[SUCCESS] Dataset is valid! (Size: {num_elements}, BlockSize: {config.BLOCK_SIZE})\033[0m")
    else:
        print(f"\033[91m[FAILURE] Dataset failed validation.\033[0m")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Set your test file name to u2
    filename = "element1024.u2"
    
    # 1. Generate any number of elements (for example 10000, not necessarily
    #    a multiple of 512). This value can be any positive integer; the script
    #    guarantees valid data.
    generate_dataset(filename, total_elements=1024)
    
    # 2. Verify
    verify_dataset(filename)