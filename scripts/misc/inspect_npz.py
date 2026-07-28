import numpy as np
files = ["pertB_data.npz","pertB_direct_data.npz","learning_curve_direct_data.npz","learning_curve_lopo_data.npz","fc_recon_teacher_sweep.npz","fc_recon_noise_sweep.npz"]
for f in files:
    try:
        d = np.load(f)
        print("=== " + f + " ===")
        for k in d.files:
            v = d[k]
            print("  " + k + ": shape=" + str(v.shape) + " dtype=" + str(v.dtype) + " min=" + str(round(float(v.min()),4)) + " max=" + str(round(float(v.max()),4)))
        print()
    except Exception as e:
        print("ERROR " + f + ": " + str(e))
