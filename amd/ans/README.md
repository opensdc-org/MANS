# AMD ANS (HIP)

HIP ANS encoder/decoder used by MANS AMD pipeline.

Typical pipeline on AMD:
1. `amd_mapping_uint16` / `amd_mapping_uint32` (ADM mapping)
2. `hipans_compress`
3. `hipans_decompress`

## Build

From repo root:

```bash
cmake -S . -B build -DTARGET_PLATFORM=amd -DBUILD_HDF5_PLUGIN=OFF
cmake --build build -j
```

If you also need CPU side tools, use `TARGET_PLATFORM=cpu_amd`.

Generated binaries:
- `build/bin/amd/amd_mapping_uint16`
- `build/bin/amd/amd_mapping_uint32`
- `build/bin/hipans_compress`
- `build/bin/hipans_decompress`

## Usage

ADM mapping:

```bash
./build/bin/amd/amd_mapping_uint16 <input file> <output file>
./build/bin/amd/amd_mapping_uint32 <input file> <output file>
```

ANS compress/decompress:

```bash
./build/bin/hipans_compress <inputfile> <tempfile>
./build/bin/hipans_decompress <input.file> <output.file>
```

Example chain (`u16`):

```bash
./build/bin/amd/amd_mapping_uint16 input_u16.bin mapped_u16.bin
./build/bin/hipans_compress mapped_u16.bin mapped_u16.ans
./build/bin/hipans_decompress mapped_u16.ans mapped_u16.restore.bin
```

## Notes

- The ANS example binaries use internal precision `10` (hardcoded in source).
- `hipans_decompress` reads metadata/header from compressed input.
- `amd/ans/CMakeLists.txt` currently pins HIP compiler paths; adjust if your ROCm installation differs.
- For full MANS CPU autotune + benchmark workflow, see root [README.md](../../README.md).
