import numpy as np
d = np.load('pertB_direct_data.npz', allow_pickle=True)
cc = d['cc_lda']; ad = d['ad_lda']
print(f"Baseline  CC={cc.mean():.3f}+-{cc.std():.3f}  AD={ad.mean():.3f}+-{ad.std():.3f}")
print(f"CC FC-r: {d['cc_fc_r'].mean():.4f}  AD base FC-r: {d['ad_fc_r_base'].mean():.4f}")
for pt in ['full_w','top5','top1']:
    alphas = d[pt+'_alphas']; lda = d[pt+'_lda']; fcr = d[pt+'_fcr']
    print(f"\n[{pt}]  CC_mean={cc.mean():.3f}")
    for ai,a in enumerate(alphas):
        row_l = lda[ai]; row_f = fcr[ai]
        print(f"  a={a:.3f}  LDA mean={row_l.mean():.3f}  std={row_l.std():.3f}  FCr={row_f.mean():.4f}")
