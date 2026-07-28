import numpy as np
d = np.load('g_space_cache.npz', allow_pickle=True)
print('G_pat_full shape:', d['G_pat_full'].shape)
print('patient_labels shape:', d['patient_labels'].shape)
print('cc_idx:', len(d['cc_idx']), 'patients')
print('ad_idx:', len(d['ad_idx']), 'patients')
print('cum_var shape:', d['cum_var'].shape)
print('cum_var[:20]:', d['cum_var'][:20])
# Also check fc_recon_noise_sweep for what sigma=0.5 gives
d2 = np.load('fc_recon_noise_sweep.npz', allow_pickle=True)
print()
print('noise_vals:', d2['noise_vals'])
print('noise_means:', d2['noise_means'])
print('r_sess range at best sigma:', d2['r_sess'].min(), '-', d2['r_sess'].max(), '  mean:', d2['r_sess'].mean())
