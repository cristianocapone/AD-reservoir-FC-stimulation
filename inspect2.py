import numpy as np
for f in ["pertB_data.npz","pertB_direct_data.npz"]:
    d = np.load(f, allow_pickle=True)
    print("=== " + f + " ===")
    for k in d.files:
        v = d[k]
        print("  " + k + ": " + str(type(v)) + " shape=" + str(getattr(v,'shape','?')) + " dtype=" + str(getattr(v,'dtype','?')))
        if hasattr(v,'dtype') and v.dtype.kind in 'biuf' and v.size < 20:
            print("    values: " + str(v))
        elif hasattr(v,'dtype') and v.dtype.kind == 'O':
            print("    sample: " + str(v.flat[0]))
    print()
