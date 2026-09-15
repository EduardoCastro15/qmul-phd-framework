from pathlib import Path
import warnings,json,sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csr_matrix,eye,diags
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components,breadth_first_order
warnings.filterwarnings('ignore',category=DeprecationWarning)
sys.path.insert(0,'docs/stats');from apply_wlnm_tukey_retention import tukey_fences
folder=Path('src/matlab/data/result_wlnm_dir_neg_roleonly_train60_trophic_v2_pilot/ecological_snapshots')
out=Path('docs/manuscript_review/2026-09-09/trophic_height_audit/lachlan_comparison')
def classic(A):
 A=csr_matrix(A);_,labels=connected_components(A,directed=False);ix=np.flatnonzero(labels==np.bincount(labels).argmax()); B=A[ix,:][:,ix];kin=np.asarray(B.sum(axis=0)).ravel();reach=set()
 for i in np.flatnonzero(kin==0):reach.update(breadth_first_order(B,int(i),directed=True,return_predecessors=False))
 if len(reach)!=len(ix):return np.nan
 t=spsolve((eye(len(ix))-diags(1/np.maximum(kin,1))@B.T).tocsc(),np.ones(len(ix)))
 return float(np.mean(t))
rows=[]
for path in folder.glob('*.mat'):
 s=loadmat(path,simplify_cells=True)['snapshot']; name=s['metadata']['Foodweb']
 d=loadmat(Path('src/matlab/data/foodwebs_mat')/(name+'.mat'),simplify_cells=True)
 role=np.asarray(d['role']).astype(str); resource=np.char.lower(role)=='resource'
 P=csr_matrix(s['pseudo']);T=csr_matrix(s['train']); E=csr_matrix(s['empirical']); modified=P.copy().tolil()
 i,j=P.nonzero();selected=(np.asarray(T[i,j]).ravel()==0)&resource[j]
 ri=i[selected];rj=j[selected];modified[ri,rj]=0
 true_removed=int(np.asarray(E[ri,rj]).sum()) if len(ri) else 0
 rows.append({'Foodweb':name,'ExperimentID':int(s['metadata']['ExperimentID']),'ResourceCount':int(sum(resource)),'RemovedPredictedLinks':len(ri),'RemovedTrueLinks':true_removed,'StrictMeanAfterResourceConstraint':classic(modified)})
df=pd.DataFrame(rows);df.to_csv(out/'exploratory_role_constraint.csv',index=False); result=[]
for name,g in df.groupby('Foodweb'):
 vals=g.StrictMeanAfterResourceConstraint.dropna().to_list();f=tukey_fences(vals,1.5);kept=[v for v in vals if f['LowerFence']<=v<=f['UpperFence']]
 result.append({'Foodweb':name,'valid':len(vals),'retained':len(kept),'total_predicted_edges_removed':int(g.RemovedPredictedLinks.sum()),'true_edges_removed':int(g.RemovedTrueLinks.sum()),'runs_changed':int((g.RemovedPredictedLinks>0).sum()),'resources':int(g.ResourceCount.iloc[0])})
ref=pd.read_csv('data/processed/Average_Trophic_Size_Height_Gateway_Dataset/mean_tl.csv').set_index('Ecosystem');sp=pd.read_csv('data/processed/Average_Trophic_Size_Height_Gateway_Dataset/species_tl.csv');exceptions=[]
for name in ['Carpinteria','Chesapeake Bay']:
 d=loadmat(Path('src/matlab/data/foodwebs_mat')/(name+'_tax_mass.mat'),simplify_cells=True);A=csr_matrix(d['net']);A.data[:]=1;A.setdiag(0);A.eliminate_zeros(); _,labs=connected_components(A,directed=False);ix=np.flatnonzero(labs==np.bincount(labs).argmax());B=A[ix,:][:,ix];kin=np.asarray(B.sum(axis=0)).ravel();t=spsolve((eye(len(ix))-diags(1/np.maximum(kin,1))@B.T).tocsc(),np.ones(len(ix)))
 mask=(t>1+1e-8)&(t<10)
 exceptions.append({'Foodweb':name,'Lachlan':float(ref.loc[name].Mean_Trophic_Level),'MeanNonbasalBelow10':float(t[mask].mean()),'NNonbasalBelow10':int(sum(mask)),'LachlanSpeciesN':int(sum(sp.Ecosystem==name)),'Above10':int(sum(t>=10))})
report={'constraint':'Remove predicted-only edges whose target role is resource. Training and empirical matrices unchanged. Exploratory only; roles loaded from full empirical data may leak held-out structure.','groups':result,'export_exceptions':exceptions}
(out/'role_constraint_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
