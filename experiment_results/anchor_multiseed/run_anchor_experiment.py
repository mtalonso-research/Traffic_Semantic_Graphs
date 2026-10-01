"""Run one controlled city UST anchor experiment with frozen validation/test sets."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from src.experiment_utils import ProjectionHead, risk_to_class_safe, set_seed
from src.graph_encoding.autoencoder import HeteroGraphAutoencoder, QuantileFeatureQuantizer, batched_graph_embeddings
from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.risk_prediction import RiskPredictionHead

def digest(v): return hashlib.sha256(''.join(f'{x}\n' for x in v).encode()).hexdigest()
def subset(root, risk, ids):
 d=get_graph_dataset(root_dir=str(root),mode='all',side_information_path=None,risk_scores_path=str(risk)); wanted=set(ids); ix=[i for i,n in enumerate(d.graph_filenames) if n.split('_')[0] in wanted]
 found=[d.graph_filenames[i].split('_')[0] for i in ix]
 if len(ix)!=len(wanted) or set(found)!=wanted: raise ValueError(f'manifest/dataset mismatch: expected {len(wanted)}, found {len(ix)}')
 return Subset(d,ix)
def paired(view, ids):
 c=subset(view/'training_data/noisy_true/graphs',view/'training_data/noisy_true/risk_scores.json',ids); n=subset(view/'training_data/noisy_0/graphs',view/'training_data/noisy_0/risk_scores.json',ids)
 ci=[c.dataset.graph_filenames[i].split('_')[0] for i in c.indices]; ni=[n.dataset.graph_filenames[i].split('_')[0] for i in n.indices]
 if ci!=ni: raise ValueError('cross-view pair order differs')
 return c,n
def score(y,p):
 cm=confusion_matrix(y,p,labels=range(4)); rec=np.diag(cm)/np.maximum(cm.sum(1),1); pre=np.diag(cm)/np.maximum(cm.sum(0),1); f=2*pre*rec/np.maximum(pre+rec,1e-12)
 return {'accuracy':float(np.trace(cm)/max(cm.sum(),1)),'balanced_accuracy':float(rec.mean()),'macro_f1':float(f.mean()),'qwk':float(cohen_kappa_score(y,p,weights='quadratic')),'class_mae':float(np.mean(np.abs(y-p))),'true_class_counts':dict(zip(map(str,range(4)),map(int,np.bincount(y,minlength=4)))),'predicted_class_counts':dict(zip(map(str,range(4)),map(int,np.bincount(p,minlength=4)))),'per_class_precision':pre.tolist(),'per_class_recall':rec.tolist(),'per_class_f1':f.tolist(),'sample_count':int(cm.sum()),'confusion_matrix':cm.tolist()}
def main(a):
 out=Path(a.output_dir); m=json.loads((Path(a.manifest_root)/'manifest.json').read_text()); view=Path(a.city_view_root)
 if m.get('protocol') not in ('controlled_city_ust_anchor_v2','scene_aware_nested_anchor_v1'): raise ValueError('unsupported manifest protocol')
 if (m['source_city'],m['target_city'])!=(a.source_city,a.target_city): raise ValueError('manifest city mismatch')
 required=('source_train','source_validation','target_train','target_validation','source_evaluation','target_evaluation')
 if any(not m[k] for k in required): raise ValueError('all training, validation, and evaluation populations must be non-empty')
 pool=list(m['target_train']); requested=round(len(pool)*a.anchor_pct/100); anchor_sha=None
 if a.anchors_file:
  ap=Path(a.anchors_file); frozen=json.loads(ap.read_text()); anchors=list(frozen['anchor_ids'][str(a.anchor_pct)]); anchor_sha=hashlib.sha256(ap.read_bytes()).hexdigest()
  if frozen.get('city')!=a.target_city or frozen.get('training_seed')!=a.seed or len(anchors)!=requested: raise ValueError('frozen anchor manifest mismatch')
 else: anchors=sorted(random.Random(m['anchor_selection_seed']).sample(pool,requested))
 if not set(anchors).issubset(set(pool)) or set(anchors)&set(m['target_validation']) or set(anchors)&set(m['target_evaluation']): raise ValueError('anchor partition integrity failure')
 if a.dry_run: print(json.dumps({'status':'dry-run-ok','requested_anchor_count':requested,'source_train':len(m['source_train']),'source_validation':len(m['source_validation']),'target_validation':len(m['target_validation'])},indent=2)); return
 print(json.dumps({'development_mode':'validation_only','test_accessed':False,'checkpoint_selection_metric':('boston_validation_macro_f1_then_target_ce' if a.selection_metric=='macro_f1' else 'boston_validation_target_ce')}),flush=True)
 out.mkdir(parents=True,exist_ok=True)
 allowed={'train.log','train.err.log','process.json'}
 if any(x.name not in allowed for x in out.iterdir()): raise FileExistsError(f'refusing to overwrite existing artifacts in {out}')
 set_seed(a.seed)
 if not a.device.startswith('cuda') or not torch.cuda.is_available(): raise RuntimeError('CUDA is required for anchor training; refusing CPU fallback')
 dev=torch.device(a.device); torch.cuda.set_device(dev)
 print(json.dumps({'device':str(dev),'gpu_name':torch.cuda.get_device_name(dev),'seed':a.seed,'anchor_pct':a.anchor_pct}),flush=True)
 st=subset(view/'training_data/clean/graphs',view/'training_data/clean/risk_scores.json',m['source_train']); sv=subset(view/'training_data/clean/graphs',view/'training_data/clean/risk_scores.json',m['source_validation']); tt=subset(view/'training_data/noisy_0/graphs',view/'training_data/noisy_0/risk_scores.json',m['target_train']); ac,an=paired(view,anchors); vc,vn=paired(view,m['target_validation'])
 meta1,meta2=st.dataset.get_metadata(),tt.dataset.get_metadata(); q1=QuantileFeatureQuantizer(bins=32,node_types=meta1[0]);q2=QuantileFeatureQuantizer(bins=32,node_types=meta2[0]); fit_n=min(512,len(st),len(tt));q1.fit(Subset(st,list(range(fit_n))));q2.fit(Subset(tt,list(range(fit_n))))
 # `weights_only` was added after the project's PyTorch 1.12 environment.
 # These are trusted local research artifacts and the default preserves the
 # original full-checkpoint loading behavior on both old and new PyTorch.
 c1=torch.load(a.ae_clean,map_location='cpu',weights_only=False);c2=torch.load(a.ae_noisy,map_location='cpu',weights_only=False);cfg=c1['args']
 # Older checkpoints can retain stale CLI embed_dim metadata.  The projection
 # tensor is the authoritative trained architecture specification.
 def checkpoint_dims(checkpoint):
  weight=checkpoint['encoder_state_dict']['lin_proj.ego.weight'];return weight.shape[1],weight.shape[0]
 hd,ed=checkpoint_dims(c1); hd2,ed2=checkpoint_dims(c2)
 if (hd,ed)!=(hd2,ed2): raise ValueError(f'AE architecture mismatch: source={(hd,ed)}, target={(hd2,ed2)}')
 def encoder(checkpoint,meta,q):
  e=HeteroGraphAutoencoder(metadata=meta,hidden_dim=hd,embed_dim=ed,quantizer_spec=q.spec(),feat_emb_dim=16,num_encoder_layers=cfg['num_encoder_layers'],num_decoder_layers=cfg['num_decoder_layers'],activation=cfg['activation'],dropout_rate=cfg['dropout_rate'],side_info_dim=0).to(dev);e.load_state_dict(checkpoint['encoder_state_dict']);e.eval();return e
 e1,e2=encoder(c1,meta1,q1),encoder(c2,meta2,q2);dim=ed*len(meta1[0]);risk=RiskPredictionHead(dim,64,4,'classification',.5).to(dev);proj=ProjectionHead(dim,dropout=.1,residual=False).to(dev)
 risk_train=a.m2_train_risk or a.m6_train_risk
 if a.risk_checkpoint:
  healthy=torch.load(a.risk_checkpoint,map_location='cpu',weights_only=False);risk.load_state_dict(healthy['state'],strict=True);risk.eval()
  if not risk_train:
   for param in risk.parameters(): param.requires_grad_(False)
 torch.save({'risk':risk.state_dict(),'projector':proj.state_dict(),'target_encoder':e2.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all(),'split_manifest_sha256':hashlib.sha256((Path(a.manifest_root)/'manifest.json').read_bytes()).hexdigest(),'anchor_manifest_sha256':hashlib.sha256(Path(a.anchors_file).read_bytes()).hexdigest() if a.anchors_file else None,'seed':a.seed,'config':vars(a)},out/'pre_ust_state.pt')
 o1=(torch.optim.AdamW(risk.parameters(),lr=(a.risk_learning_rate if risk_train else a.learning_rate),weight_decay=a.weight_decay) if (not a.risk_checkpoint or risk_train) else None);o2=torch.optim.AdamW(list(proj.parameters())+(list(e2.parameters()) if (a.m4_train_target_encoder or a.m6_train_risk) else [])+(list(risk.parameters()) if a.m6_train_risk else []),lr=a.learning_rate,weight_decay=a.weight_decay);ce=torch.nn.CrossEntropyLoss()
 def emb(b,e,q,meta):
  b=q.transform_inplace(b).to(dev);z,_,_=e(b);return b,batched_graph_embeddings(z,b,meta,embed_dim_per_type=ed)
 def validation():
  # Checkpoint selection must use the same deterministic inference modes as
  # final validation evaluation; M8 has a trainable target encoder.
  e1.eval();e2.eval();risk.eval();proj.eval();rs=0.;rn=0;als=0.;aln=0;ts=0.;tn=0;ys=[];ps=[]
  with torch.no_grad():
   for b in DataLoader(sv,batch_size=a.batch_size,shuffle=False,num_workers=0):
    b,g=emb(b,e1,q1,meta1);rs+=ce(risk(g),risk_to_class_safe(b.y)).item()*b.num_graphs;rn+=b.num_graphs
   for bc,bn in zip(DataLoader(vc,batch_size=a.batch_size,shuffle=False,num_workers=0),DataLoader(vn,batch_size=a.batch_size,shuffle=False,num_workers=0)):
    _,gc=emb(bc,e1,q1,meta1);bn,gn=emb(bn,e2,q2,meta2);als+=F.mse_loss(proj(gn),gc).item();aln+=1;logits=risk(proj(gn));ts+=ce(logits,risk_to_class_safe(bn.y)).item()*bn.num_graphs;tn+=bn.num_graphs;ys.extend(risk_to_class_safe(bn.y).cpu().numpy());ps.extend(logits.argmax(1).cpu().numpy())
  sr=rs/max(rn,1);al=als/max(aln,1);return ts/max(tn,1),sr,al,score(np.asarray(ys),np.asarray(ps))['macro_f1']
 sl=DataLoader(st,batch_size=a.batch_size,shuffle=True,num_workers=0);cl=DataLoader(ac,batch_size=a.batch_size,shuffle=True,num_workers=0);nl=DataLoader(an,batch_size=a.batch_size,shuffle=True,num_workers=0);source_steps=len(sl);natural_adaptation_steps=len(cl);adaptation_steps=a.fixed_target_updates_per_epoch or a.adaptation_steps_per_epoch or natural_adaptation_steps
 if adaptation_steps < 1: raise ValueError('adaptation_steps_per_epoch must be positive')
 source_replay_steps=a.source_replay_steps or source_steps
 source_class_weight=None
 if risk_train:
  source_labels=np.asarray([int(risk_to_class_safe(st[i].y).item()) for i in range(len(st))])
  counts=np.bincount(source_labels,minlength=4).astype(np.float32)
  source_class_weight=torch.tensor(counts.sum()/np.maximum(counts,1)/4,dtype=torch.float32,device=dev)
 anchor_labels=np.asarray([int(risk_to_class_safe(an[i].y).item()) for i in range(len(an))]) if a.paired_anchor_batches else None
 anchor_probs=None
 if a.class_balanced_anchors:
  if anchor_labels is None: raise ValueError('class-balanced anchors require paired-anchor-batches')
  counts=np.bincount(anchor_labels,minlength=4).astype(np.float64)
  anchor_probs=(1.0/np.maximum(counts[anchor_labels],1));anchor_probs/=anchor_probs.sum()
 anchor_class_weight=None
 if a.anchor_weighted_ce:
  if anchor_labels is None: raise ValueError('anchor-weighted CE requires paired anchors')
  counts=np.bincount(anchor_labels,minlength=4).astype(np.float32)
  anchor_class_weight=torch.tensor(counts.sum()/np.maximum(counts,1)/4,dtype=torch.float32,device=dev)
 source_prototypes=None
 if a.m5_class_conditional:
  sums=torch.zeros(4,dim,device=dev); nums=torch.zeros(4,device=dev)
  e1.eval()
  with torch.no_grad():
   for sb in DataLoader(st,batch_size=a.batch_size,shuffle=False,num_workers=0):
    sb,sg=emb(sb,e1,q1,meta1); sy=risk_to_class_safe(sb.y)
    for klass in range(4):
     mask=sy==klass
     if mask.any(): sums[klass]+=sg[mask].sum(0);nums[klass]+=mask.sum()
  if (nums==0).any(): raise ValueError('M5 source prototype missing a class')
  source_prototypes=sums/nums[:,None]
 print(json.dumps({'source_steps_per_epoch':source_steps,'source_replay_steps_per_epoch':source_replay_steps,'natural_adaptation_steps_per_epoch':natural_adaptation_steps,'actual_adaptation_steps_per_epoch':adaptation_steps,'controlled_adaptation_steps':a.adaptation_steps_per_epoch is not None or a.fixed_target_updates_per_epoch is not None,'fixed_target_updates_per_epoch':a.fixed_target_updates_per_epoch,'m2_train_risk':a.m2_train_risk,'m4_train_target_encoder':a.m4_train_target_encoder,'m5_class_conditional':a.m5_class_conditional,'paired_anchor_batches':a.paired_anchor_batches,'class_balanced_anchors':a.class_balanced_anchors}),flush=True)
 # For Macro-F1 selection the sentinel must be -infinity, not NaN: every
 # measured validation score must be eligible to become the first checkpoint.
 best=(float('inf'),None,0,float('-inf')); stale=0
 for epoch in range(1,a.epochs+1):
  if o1:
   risk.train()
   si=iter(sl)
   for _ in range(source_replay_steps):
    try: b=next(si)
    except StopIteration: si=iter(sl);b=next(si)
    o1.zero_grad();b,g=emb(b,e1,q1,meta1);v=F.cross_entropy(risk(g),risk_to_class_safe(b.y),weight=source_class_weight);v.backward();o1.step()
  proj.train();e2.train() if (a.m4_train_target_encoder or a.m6_train_risk) else e2.eval();ci=iter(cl);ni=iter(nl);full_perm=np.random.permutation(len(ac)) if a.full_anchor_epochs else None
  for step in range(adaptation_steps):
   if a.paired_anchor_batches:
    if a.full_anchor_epochs:
     # M8 fixed exposure: complete one shuffled full pass, then cycle that
     # same train-only permutation only to meet the shared update budget.
     positions=np.arange(step*a.batch_size,(step+1)*a.batch_size)
     ix=full_perm[positions % len(ac)] if a.fixed_target_updates_per_epoch else full_perm[positions[positions<len(ac)]]
    elif a.class_balanced_anchors:
     ix=np.random.choice(len(ac),size=a.batch_size,replace=True,p=anchor_probs)
    else:
     ix=np.random.choice(len(ac),size=a.batch_size,replace=len(ac)<a.batch_size)
    bc=Batch.from_data_list([ac[int(i)] for i in ix]);bn=Batch.from_data_list([an[int(i)] for i in ix])
   else:
    try: bc=next(ci);bn=next(ni)
    except StopIteration:
     ci=iter(cl);ni=iter(nl);bc=next(ci);bn=next(ni)
   o2.zero_grad();_,gc=emb(bc,e1,q1,meta1);bn,gn=emb(bn,e2,q2,meta2);ty=risk_to_class_safe(bn.y);aligned=F.mse_loss(proj(gn),source_prototypes[ty]) if a.m5_class_conditional else F.mse_loss(proj(gn),gc);v=aligned+a.lambda_anchor*F.cross_entropy(risk(proj(gn)),ty,weight=anchor_class_weight);v.backward();o2.step()
  total,source_v,align_v,macro_f1=validation();print(json.dumps({'epoch':epoch,'target_validation_classification_loss':total,'target_validation_macro_f1':macro_f1,'source_validation_loss':source_v,'paired_validation_alignment_loss':align_v}),flush=True)
  criterion=(total,) if a.selection_metric=='target_ce' else (-macro_f1,total)
  best_criterion=(best[0],) if a.selection_metric=='target_ce' else (-best[3],best[0])
  if criterion<best_criterion: best=(total,{'risk':{k:v.detach().cpu() for k,v in risk.state_dict().items()},'projector':{k:v.detach().cpu() for k,v in proj.state_dict().items()},'target_encoder':{k:v.detach().cpu() for k,v in e2.state_dict().items() if not isinstance(v,torch.nn.parameter.UninitializedParameter)}},epoch,macro_f1);stale=0
  else:
   stale+=1
   if stale>=a.patience: print(json.dumps({'early_stopping':True,'epoch':epoch,'best_epoch':best[2]}),flush=True);break
 risk.load_state_dict(best[1]['risk']);proj.load_state_dict(best[1]['projector']);e2.load_state_dict(best[1]['target_encoder'],strict=False);e2.eval()
 def evaluate(*_args,**_kwargs):
  raise RuntimeError('Hard guard: evaluation/test splits are forbidden in validation-only development.')
 # Development runner: never load either frozen evaluation/test population.
 def evaluate_target_validation():
  y=[];p=[];risk.eval();proj.eval()
  with torch.no_grad():
   for b in DataLoader(vn,batch_size=a.batch_size,shuffle=False,num_workers=0):
    b,g=emb(b,e2,q2,meta2);y.extend(risk_to_class_safe(b.y).cpu().numpy());p.extend(risk(proj(g)).argmax(1).detach().cpu().numpy())
  return score(np.asarray(y),np.asarray(p))
 t=evaluate_target_validation();s=None
 protocol='m8_fixed_target_exposure_validation_f1_v1' if a.m8_fixed_exposure else ('m6_target_representation_classifier_replay_validation_f1_v1' if a.m6_train_risk else ('m2_boston_validation_only_v1' if a.m2_train_risk else ('m5_class_conditional_prototype_validation_f1_v1' if a.m5_class_conditional else ('m4_target_representation_validation_f1_v1' if a.m4_train_target_encoder else ('m3_projector_balanced_validation_f1_v1' if a.selection_metric=='macro_f1' else 'm1_boston_validation_only_v3')))))
 result={'protocol':protocol,'checkpoint_selection_metric':('boston_validation_macro_f1_then_target_ce' if a.selection_metric=='macro_f1' else 'boston_validation_target_ce'),'source_city':a.source_city,'target_city':a.target_city,'split_seed':m['split_seed'],'training_seed':a.seed,'seed':a.seed,'anchor_pct':a.anchor_pct,'requested_anchor_count':requested,'actual_anchor_count':len(anchors),'best_validation_target_ce':best[0],'best_validation_macro_f1':best[3],'best_epoch':best[2],'metrics':{'boston_validation':t},'test_accessed':False}
 torch.save({'state':best[1],'quantizers':{'source':q1,'target':q2},'manifest':m,'result':result,'ae_paths':[a.ae_clean,a.ae_noisy]},out/'best_model.pt');(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':'completed','best_epoch':best[2]}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-city',default='singapore');p.add_argument('--target-city',required=True);p.add_argument('--city-view-root',required=True);p.add_argument('--ae-clean',required=True);p.add_argument('--ae-noisy',required=True);p.add_argument('--manifest-root',required=True);p.add_argument('--anchors-file');p.add_argument('--anchor-pct',type=int,choices=[3,5,10,20,50],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output-dir',required=True);p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=20);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--adaptation-steps-per-epoch',type=int,default=None);p.add_argument('--fixed-target-updates-per-epoch',type=int,default=None);p.add_argument('--risk-checkpoint');p.add_argument('--lambda-anchor',type=float,default=1.0);p.add_argument('--selection-metric',choices=['target_ce','macro_f1'],default='target_ce');p.add_argument('--m2-train-risk',action='store_true');p.add_argument('--m4-train-target-encoder',action='store_true');p.add_argument('--m5-class-conditional',action='store_true');p.add_argument('--m6-train-risk',action='store_true');p.add_argument('--m8-fixed-exposure',action='store_true');p.add_argument('--full-anchor-epochs',action='store_true');p.add_argument('--anchor-weighted-ce',action='store_true');p.add_argument('--risk-learning-rate',type=float,default=1e-5);p.add_argument('--source-replay-steps',type=int,default=None);p.add_argument('--paired-anchor-batches',action='store_true');p.add_argument('--class-balanced-anchors',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--dry-run',action='store_true');main(p.parse_args())
