from pathlib import Path
import warnings,json
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix,diags,eye
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components,breadth_first_order
from scipy.stats import spearmanr
warnings.filterwarnings('ignore',category=DeprecationWarning)
out=Path('docs/manuscript_review/2026-09-09/trophic_height_audit/lachlan_comparison')
ref=pd.read_csv('data/processed/Average_Trophic_Size_Height_Gateway_Dataset/mean_tl.csv').set_index('Ecosystem')
sp=pd.read_csv('data/processed/Average_Trophic_Size_Height_Gateway_Dataset/species_tl.csv')
rows=[]
for file in sorted(Path('src/matlab/data/foodwebs_mat').glob('*.mat')):
 name=file.stem.removesuffix('_tax_mass'); data=loadmat(file,simplify_cells=True)
 A=csr_matrix(data['net']);A.data[:]=1; A.setdiag(0); A.eliminate_zeros()
 _,lab=connected_components(A,directed=False); ix=np.flatnonzero(lab==np.bincount(lab).argmax()); B=A[ix,:][:,ix]
 kin=np.asarray(B.sum(axis=0)).ravel();kout=np.asarray(B.sum(axis=1)).ravel()
 L=diags(kin+kout)-B-B.T;h=np.zeros(len(ix));h[1:]=spsolve(L[1:,1:].tocsc(),kin[1:]-kout[1:]);h-=h.min()
 reached=set()
 for basal in np.flatnonzero(kin==0):reached.update(breadth_first_order(B,int(basal),directed=True,return_predecessors=False))
 t=np.full(len(ix),np.nan)
 if len(reached)==len(ix): t=spsolve((eye(len(ix))-diags(1/np.maximum(kin,1))@B.T).tocsc(),np.ones(len(ix)))
 r={'Foodweb':name,'FullN':A.shape[0],'LccN':len(ix),'Classic':float(t.mean()),'Generalized':float(h.mean()),'GeneralizedPlus1':float(h.mean()+1),'BasalCount':int((kin==0).sum())}
 if name in ref.index:
  lr=ref.loc[name];r.update({'Lachlan':float(lr.Mean_Trophic_Level),'LachlanFullN':int(lr.Num_Species_full),'LachlanLccN':int(lr.Num_Species_largest)})
 ss=sp[sp.Ecosystem==name]
 r['SpeciesCSVN']=len(ss);r['SpeciesCSVMean']=ss.Trophic_level.mean();r['SpeciesCSVBasalN']=int(np.isclose(ss.Trophic_level,1).sum())
 r['ClassicNonBasal']=t[t>1+1e-8].mean() if (t>1+1e-8).any() else np.nan
 rows.append(r)
df=pd.DataFrame(rows);df.to_csv(out/'empirical_comparison.csv',index=False)
summary={'matrices':len(df),'lachlan_rows':len(ref),'matched':int(df.Lachlan.notna().sum()),'missing_references':df.loc[df.Lachlan.isna(),'Foodweb'].tolist(),'classic_invalid':df.loc[~np.isfinite(df.Classic),'Foodweb'].tolist(),'size_mismatches':df.loc[(df.Lachlan.notna())&((df.FullN!=df.LachlanFullN)|(df.LccN!=df.LachlanLccN)),['Foodweb','FullN','LachlanFullN','LccN','LachlanLccN']].to_dict('records')}
for col in ['Classic','Generalized','GeneralizedPlus1','SpeciesCSVMean','ClassicNonBasal']:
 d=df[np.isfinite(df[col])&np.isfinite(df.Lachlan)];delta=d[col]-d.Lachlan;rel=delta.abs()/d.Lachlan
 summary[col]={'n':len(d),'matches_1e-8':int(np.isclose(d[col],d.Lachlan,atol=1e-8,rtol=0).sum()),'mean_absolute_difference':float(delta.abs().mean()),'median_absolute_difference':float(delta.abs().median()),'mean_signed_difference':float(delta.mean()),'median_absolute_percent_difference':float(100*rel.median()),'max_absolute_difference':float(delta.abs().max()),'over_10_percent':int((rel>.1).sum()),'spearman':float(spearmanr(d[col],d.Lachlan).statistic)}
summary['focus']=df[df.Foodweb.isin(['Ythan Estuary','Dutch Microfauna food web PlotB','Blackrock Stream','Sutton Stream','Gearagh','Weddell Sea'])].to_dict('records')
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True)+'\n'); print(json.dumps(summary,indent=2,allow_nan=True))
