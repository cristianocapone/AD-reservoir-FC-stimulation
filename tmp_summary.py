import numpy as np
d = np.load('pertB_direct_data.npz', allow_pickle=True)
alphas = d['top5_alphas']
lda    = d['top5_lda']
fcr    = d['top5_fcr']
cc_lda = d['cc_lda']
ad_lda = d['ad_lda']
thr    = 0.5*(cc_lda.mean() + ad_lda.mean())
gap    = ad_lda.mean() - cc_lda.mean()

print("alpha  LDA_mean  LDA_std  FC-r  gap_closed%  reclassif%")
print("-"*62)
for i, a in enumerate(alphas):
    m    = lda[i].mean()
    s    = lda[i].std()
    fr   = fcr[i].mean()
    gp   = (ad_lda.mean() - m) / gap * 100
    recl = (lda[i] < thr).mean() * 100
    flag = " <<< UNPHYSIO (RMS>3x)" if a >= 5.0 else ""
    print(f"a={a:5.2f}  {m:+7.3f}  {s:.3f}   {fr:.3f}  {gp:5.1f}%       {recl:4.1f}%{flag}")

print()
print(f"CC mean = {cc_lda.mean():.3f}   AD mean = {ad_lda.mean():.3f}   midpoint = {thr:.3f}")
print(f"Total gap = {gap:.3f}")
print()
print("Physiological boundary: RMS_top5 grows ~1.6x per alpha unit above 2")
print("  a=3: RMS ~1.6x (marginal)  a=4: ~2.5x  a=5: ~3.4x (unphysiological)")
print("  a=15: RMS ~12x  a=20: ~16x (extreme extrapolation)")
