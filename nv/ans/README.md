# NVIDIA ANS (CUDA)

CUDA ANS encoder/decoder used by MANS NVIDIA pipeline.

Typical pipeline on NVIDIA:
1. `nv_mapping_uint16` / `nv_mapping_uint32` (ADM mapping)
2. `cudaans_compress`
3. `cudaans_decompress`

## Build

From repo root:

```bash
cmake -S . -B build -DTARGET_PLATFORM=nv -DBUILD_HDF5_PLUGIN=OFF
cmake --build build -j
```

If you also need CPU side tools, use `TARGET_PLATFORM=cpu_nv`.

Generated binaries:
- `build/bin/nv/nv_mapping_uint16`
- `build/bin/nv/nv_mapping_uint32`
- `build/bin/nv/cudaans_compress`
- `build/bin/nv/cudaans_decompress`

## Usage

ADM mapping:

```bash
./build/bin/nv/nv_mapping_uint16 <input file> <output file>
./build/bin/nv/nv_mapping_uint32 <input file> <output file>
```

ANS compress/decompress:

```bash
./build/bin/nv/cudaans_compress <inputfile> <tempfile>
./build/bin/nv/cudaans_decompress <input.file> <output.file>
```

Example chain (`u16`):

```bash
./build/bin/nv/nv_mapping_uint16 input_u16.bin mapped_u16.bin
./build/bin/nv/cudaans_compress mapped_u16.bin mapped_u16.ans
./build/bin/nv/cudaans_decompress mapped_u16.ans mapped_u16.restore.bin
```

## Notes

- The ANS example binaries use internal precision `10` (hardcoded in source).
- `cudaans_decompress` reads metadata/header from compressed input.
- For full MANS CPU autotune + benchmark workflow, see root [README.md](../../README.md).
