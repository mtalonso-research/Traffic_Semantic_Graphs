"""Train one AE using only a frozen train partition; select by frozen validation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from src.experiment_utils import set_seed
from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import HeteroGraphAutoencoder,QuantileFeatureQuantizer,feature_loss,edge_loss,batched_graph_embeddings

def subset(root,risk,ids):
 d=get_graph_dataset(root_dir=str(root),mode='all',side_information_path=None,risk_scores_path=str(risk)); wanted=set(ids); ix=[i for i,n in enumerate(d.graph_filenames) if n.split('_')[0] in wanted]
 if len(ix)!=len(wanted): raise ValueError('manifest/dataset mismatch')
 return Subset(d,ix)
def main(a):
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists(): raise FileExistsError(out)
 m=json.loads(Path(a.manifest).read_text()); ids=m[f'{a.domain}_train']; valids=m[f'{a.domain}_validation']; view=Path(a.view)
 alias='clean' if a.domain=='source' else 'noisy_0'; risk=view/'training_data'/alias/'risk_scores.json'; graphs=view/'training_data'/alias/'graphs'
 tr=subset(graphs,risk,ids); va=subset(graphs,risk,valids); meta=tr.dataset.get_metadata(); q=QuantileFeatureQuantizer(bins=32,node_types=meta[0]); q.fit(tr)
 set_seed(a.seed); dev=torch.device('cuda:0'); model=HeteroGraphAutoencoder(metadata=meta,hidden_dim=64,embed_dim=64,quantizer_spec=q.spec(),feat_emb_dim=16,num_encoder_layers=1,num_decoder_layers=1,activation='relu',dropout_rate=.1,side_info_dim=0).to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
 loaders=(DataLoader(tr,batch_size=64,shuffle=True,num_workers=0),DataLoader(va,batch_size=64,shuffle=False,num_workers=0)); best=float('inf'); bestepoch=0; stale=0
 for epoch in range(1,11):
  values=[]
  for train,loader in zip((True,False),loaders):
   model.train(train); total=0.; count=0
   with torch.set_grad_enabled(train):
    for b in loader:
     if train: opt.zero_grad()
     b=q.transform_inplace(b).to(dev); z,f,e=model(b); loss=feature_loss(f,b)+edge_loss(e,z,model.edge_decoders,num_neg=1 if train else 4)['total']
     if train: loss.backward(); opt.step()
     total+=loss.item(); count+=1
   values.append(total/max(count,1))
  print(json.dumps({'epoch':epoch,'train_reconstruction_loss':values[0],'validation_reconstruction_loss':values[1]}),flush=True)
  if values[1]<best:
   best=values[1]; bestepoch=epoch; stale=0; torch.save({'stage':'autoencoder_best','encoder_state_dict':model.state_dict(),'best_ae_val_recon':best,'args':{'num_encoder_layers':1,'num_decoder_layers':1,'activation':'relu','dropout_rate':.1},'frozen_manifest_sha256':__import__('hashlib').sha256(Path(a.manifest).read_bytes()).hexdigest(),'domain':a.domain,'train_count':len(tr),'validation_count':len(va)},out)
  else:
   stale+=1
   if stale>=10: break
 print(json.dumps({'status':'completed','best_epoch':bestepoch,'best_validation_reconstruction_loss':best}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--view',required=True);p.add_argument('--domain',choices=['source','target'],required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);main(p.parse_args())
