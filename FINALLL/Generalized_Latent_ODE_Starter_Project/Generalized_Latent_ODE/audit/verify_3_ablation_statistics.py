import json, numpy as np
from scipy import stats
J=json.load(open('results/ablation_2x2/ablation_2x2_results.json')); Z=np.load('results/ablation_2x2/predictions_test.npz')
print('JSON top-level keys:',list(J.keys()))
qt=Z['query_t']; truth=Z['truth']; obs_t=Z['obs_t']; fam=Z['family']; fut=qt>7
# observation mask from observed timestamps
idx=np.abs(obs_t[:,:,None]-qt[None,None,:]).argmin(-1); assert np.allclose(qt[idx],obs_t,atol=1e-4)
mask=np.zeros(truth.shape[:2],bool); np.put_along_axis(mask,idx,True,1)
print('observed times: n/traj',np.unique(mask.sum(1)),'max obs time',obs_t.max(),'all <=7:',bool((obs_t<=7).all()),'| first obs at t=0 for all:',bool((obs_t[:,0]==0).all()))
miss=(~mask)&(~fut)[None]
def per_traj(pred):
    d2=((pred-truth)**2).sum(-1)
    return {'recon_rmse':np.sqrt((d2*miss).sum(1)/(2*miss.sum(1))),'forecast_rmse':np.sqrt(d2[:,fut].sum(1)/(2*fut.sum()))}
models=['A','B','C','D','orig_baseline','orig_medium','orig_large']
P={m:per_traj(Z['pred_'+m]) for m in models}
FAMS=['circle','spiral','ellipse','figure_eight','sinusoid']; sel={f:fam==i for i,f in enumerate(FAMS)}; sel['overall']=np.ones(len(fam),bool)
bad=0; worst=0
for m in models:
    for g,s in sel.items():
        for k in ('recon_rmse','forecast_rmse'):
            r=J['results'][m][g][k]; v=P[m][k][s]
            for nm,val in (('mean',v.mean()),('median',np.median(v)),('sd',v.std(ddof=1))):
                e=abs(val-r[nm]); worst=max(worst,e); bad+=e>1e-5
print(f'[1] recomputed mean/median/SD of per-trajectory RMSE vs JSON: {bad} mismatches > 1e-5 (worst abs diff {worst:.2e}) over {len(models)*6*2*3} values')
# pooled
bad=0
for m in models:
    d2=((Z['pred_'+m]-truth)**2).sum(-1); pf=np.sqrt(d2[:,fut].mean()/2); pr=np.sqrt(d2[miss].sum()/(2*miss.sum()))
    bad+= abs(pf-J['pooled_rmse'][m]['overall']['forecast_rmse'])>1e-5 or abs(pr-J['pooled_rmse'][m]['overall']['recon_rmse'])>1e-5
print(f'[2] pooled overall RMSE mismatches: {bad}')
# factorial effects
A,B,C,D=(P[c] for c in 'ABCD')
def contr(k,s):
    a,b,c,d=A[k][s],B[k][s],C[k][s],D[k][s]
    return {'latent_main_effect (16 vs 8, averaged over GRU)':.5*((b-a)+(d-c)),'gru_main_effect (128 vs 64, averaged over latent)':.5*((c-a)+(d-b)),'interaction ((D-C) - (B-A))':(d-c)-(b-a)}
rng=np.random.default_rng(777); rows=[]
for g,s in sel.items():
    for k in ('recon_rmse','forecast_rmse'):
        for c,d in contr(k,s).items():
            key=f'{g}/{k}/{c}'; j=J['factorial'][key]
            bm=d[rng.integers(0,len(d),(20000,len(d)))].mean(1); lo,hi=np.percentile(bm,[2.5,97.5])
            p=stats.wilcoxon(d).pvalue; rows.append((key,p,d.mean(),lo,hi,j))
print(f'[3] factorial tests in JSON: {len(J["factorial"])} (recomputed {len(rows)}); report text claims 90')
ps=np.array([r[1] for r in rows]); order=np.argsort(ps); adj=np.empty(len(ps)); run=0
for rk,i in enumerate(order): run=max(run,min(1,(len(ps)-rk)*ps[i])); adj[i]=run
mm=cm=pm=0; wd=wc=0
for (key,p,mean,lo,hi,j),a in zip(rows,adj):
    mm+=abs(mean-j['mean'])>1e-7; wd=max(wd,abs(lo-j['bootstrap_ci95'][0]),abs(hi-j['bootstrap_ci95'][1]))
    pm+= not np.isclose(a,j['wilcoxon_p_holm'],rtol=1e-6,atol=1e-300)
print(f'    effect means mismatching: {mm}; max CI endpoint difference (independent bootstrap, 20k vs 10k resamples, different seed): {wd:.4f}; Holm-adjusted p mismatches: {pm}')
print('[4] overall effects recomputed:')
for key,p,mean,lo,hi,j in rows:
    if key.startswith('overall/'): print(f'    {key:75s} {mean:+.4f} [{lo:+.4f},{hi:+.4f}]  JSON {j["mean"]:+.4f} holm p {j["wilcoxon_p_holm"]:.2g}')
# does each effect sign pattern match the claimed finding?  per cell overall means
print('[5] overall per-cell means (recomputed):')
for m in 'ABCD': print(f'    {m}: recon {P[m]["recon_rmse"].mean():.4f}  forecast {P[m]["forecast_rmse"].mean():.4f}')
print('    vs A:', {m:(round(P[m]['recon_rmse'].mean()/A['recon_rmse'].mean()-1,3),round(P[m]['forecast_rmse'].mean()/A['forecast_rmse'].mean()-1,3)) for m in 'BCD'})
# per-family sign consistency of recon/forecast change vs A
for m in 'BCD':
    print(f'    {m}: families where forecast mean is HIGHER than A:',[f for f in FAMS if P[m]['forecast_rmse'][sel[f]].mean()>A['forecast_rmse'][sel[f]].mean()],'| recon LOWER than A:',[f for f in FAMS if P[m]['recon_rmse'][sel[f]].mean()<A['recon_rmse'][sel[f]].mean()])
# claims in ABLATION_2X2_REPORT.md
print('[6] report claim check (claimed vs recomputed):')
claim={'A':(0.0796,0.3998),'B':(0.0533,0.5199),'C':(0.0563,0.4248),'D':(0.0404,0.5936)}
for m,(cr,cf) in claim.items(): print(f'    cell {m} overall recon claimed {cr:.4f} actual {P[m]["recon_rmse"].mean():.4f} | forecast claimed {cf:.4f} actual {P[m]["forecast_rmse"].mean():.4f}')
print('    claimed recon change vs A: B -33.0%, C -29.3%, D -49.2% ; actual:',{m:f'{(P[m]["recon_rmse"].mean()/A["recon_rmse"].mean()-1)*100:+.1f}%' for m in 'BCD'})
print('    report says per-trajectory RMSE for all test trajectories are stored in the JSON:',any(isinstance(v,list) and len(v)==750 for v in J.values()),'(no per-trajectory arrays at top level; keys above)')
print('    per-family forecast (report lists circle .27/.38/.29/.52 spiral .49/.54/.50/.66 ellipse .39/.37/.33/.52 fig8 .47/.45/.43/.48 sin .37/.86/.58/.78):')
for f in FAMS: print('     ',f,[round(P[m]['forecast_rmse'][sel[f]].mean(),2) for m in 'ABCD'])
