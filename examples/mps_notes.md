# MPS (Apple Silicon) Compatibility Notes

When using Jojo on Apple Silicon (M1/M2/M3) Macs, you can leverage the Metal Performance Shaders (MPS) backend for PyTorch to accelerate model inference.

## Setup for MPS

1. Ensure you're using PyTorch 2.0+ which has stable MPS support
2. Install PyTorch with MPS support:
   ```bash
   pip install torch torchvision
   ```

## Using MPS with Jojo

To use the MPS backend, specify it as the device:

```bash
python gen.py --device mps
```

Or let Jojo auto-detect it:

```bash
python gen.py
```

## Troubleshooting

If you encounter MPS-related errors:

1. **"MPS backend not available"**:
   - Verify you have PyTorch 2.0+ installed
   - Verify you're on macOS 12.3+ with Apple Silicon

2. **Memory-related errors**:
   - Try using a smaller model
   - Reduce the context length
   - Use `float16` precision: `python gen.py --device mps --dtype float16`

3. **Operation not implemented errors**:
   - Some operations may not be supported in the MPS backend
   - Force CPU usage if needed: `python gen.py --device cpu`

## Performance Notes

- MPS provides significant speedup compared to CPU on Apple Silicon
- For best performance, use `bfloat16` or `float16` precision
- The first generation run may be slower due to compilation

## Checking MPS Availability

To verify MPS is available, you can run this Python code:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```
