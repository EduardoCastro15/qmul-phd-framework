import sys,csv,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,'docs/stats')
from apply_wlnm_tukey_retention import tukey_fences
root=Path('src/matlab/data/result_wlnm_dir_neg_roleonly_train60_trophic_v2_pilot')
groups={}; maximum_residual=0
for path in (root/'ecological_snapshots').glob('*.mat'):
 snap=loadmat(path,simplify_cells=True)['snapshot']; vals={}
 for name in ['empirical','train','pseudo']:
  A=csr_matrix(snap[name]); _,labels=connected_components(A,directed=False)
  idx=np.flatnonzero(labels==np.bincount(labels).argmax()); B=A[idx,:][:,idx].toarray()
  kin=B.sum(axis=0); kout=B.sum(axis=1)
  L=np.diag(kin+kout)-B-B.T; rhs=kin-kout
  h=np.zeros(len(idx)); h[1:]=np.linalg.solve(L[1:,1:],rhs[1:])
  residual=np.linalg.norm(L@h-rhs,1)/(np.linalg.norm(L,1)*np.linalg.norm(h,1)+np.linalg.norm(rhs,1)+np.finfo(float).eps)
  assert residual<1e-12 and np.isfinite(h).all()
  maximum_residual=max(maximum_residual,residual)
  h-=h.min()
  vals[name]=float(h.mean())
 groups.setdefault(snap['metadata']['Foodweb'],[]).append(vals)
report={'definition':'Lh=kin-kout; min(h)=0 on same largest weak component; mean(h), no +1 shift','maximum_scaled_residual':maximum_residual,'groups':[]}
for name,rows in sorted(groups.items()):
 values=[r['pseudo'] for r in rows]; f=tukey_fences(values,1.5)
 kept=[v for v in values if f['LowerFence']<=v<=f['UpperFence']]
 report['groups'].append({'Foodweb':name,'n':len(rows),'finite':len(values),'retained':len(kept),'mean_after_tukey':float(np.mean(kept)),'empirical':rows[0]['empirical'], 'min':min(values),'max':max(values)})
print(json.dumps(report,indent=2))
Path('/tmp/generalized_trophic_probe.json').write_text(json.dumps(report,indent=2))
