"""Run a real matmul on the actual NPU, and compare against OpenVINO-CPU.

This gives the honest three-way comparison the review asked for:
  our VEX-256 kernel  vs  OpenVINO CPU (oneDNN)  vs  NPU.

The NPU will not accept GGUF quant formats, so this is FP16 with OpenVINO's own
weight handling. It answers one question: for the shapes we care about
(M=1 decode GEMV, M=128 prefill GEMM), is the NPU even competitive?
"""
import time
import numpy as np
import openvino as ov
from openvino import opset13 as ops

np.random.seed(0)

def build(M, K, N, dt):
    A = ops.parameter([M, K], dt, name="A")
    Wt = (np.random.randn(K, N) * 0.02).astype(np.float32)
    if dt == ov.Type.f16:
        Wt = Wt.astype(np.float16)
    W = ops.constant(Wt)
    # matmul(a, b, transpose_a, transpose_b) - the two bools are required
    return ov.Model([ops.matmul(A, W, False, False)], [A], "matmul")

core = ov.Core()
print("openvino", ov.__version__)
print("devices", core.available_devices)
for d in core.available_devices:
    try:
        print(" ", d, core.get_property(d, "FULL_DEVICE_NAME"))
    except Exception as e:
        print(" ", d, "prop err", e)
print()

shapes = [(1, 2048, 2048), (32, 2048, 2048), (128, 2048, 2048)]
devs = ["CPU", "NPU", "GPU"]

print("%-6s %-6s %-5s %10s %11s %12s" % ("M", "dev", "dt", "compile s", "infer ms", "GMAC/s"))
print("-" * 62)
for (M, K, N) in shapes:
    for dev in devs:
        for dt, nm in ((ov.Type.f16, "f16"), (ov.Type.f32, "f32")):
            if dev == "NPU" and dt == ov.Type.f32:
                continue  # NPU is an FP16 machine; try it anyway below
            try:
                model = build(M, K, N, dt)
                t0 = time.time()
                cm = core.compile_model(model, dev)
                t1 = time.time()
                req = cm.create_infer_request()
                x = np.random.randn(M, K).astype(np.float16 if dt == ov.Type.f16 else np.float32)
                req.infer({0: x})
                reps = 50 if M == 1 else 10
                t2 = time.time()
                for _ in range(reps):
                    req.infer({0: x})
                t3 = time.time()
                ms = (t3 - t2) / reps * 1000.0
                print("%-6d %-6s %-5s %10.2f %11.3f %12.1f" %
                      (M, dev, nm, t1 - t0, ms, M * K * N / (ms / 1e3) / 1e9))
            except Exception as e:
                msg = str(e).splitlines()[0][:70]
                print("%-6d %-6s %-5s %10s %11s   FAILED: %s" % (M, dev, nm, "-", "-", msg))
    print()
