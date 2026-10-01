"""M8 Pittsburgh train-label-only class-stratified strict-nested anchor manifests."""
from pathlib import Path
import hashlib,json,random
ROOT=Path(__file__).resolve().parents[2];F=ROOT/'experiment_results/nested_scene_split/frozen_pittsburgh';M=F/'manifests/singapore_to_pittsburgh/manifest.json';R=F/'city_views/singapore_to_pittsburgh/training_data/noisy_0/risk_scores.json';O=ROOT/'experiment_results/adaptation_repair/m8_stratified_pittsburgh_anchors';C={3:168,5:281,10:561,20:1122,50:2806}
def cl(v):return 3 if v>.3442 else 2 if v>.1008 else 1 if v>.0043 else 0
def dg(x):return hashlib.sha256(x).hexdigest()
def order(b,seed):
 s={k:v[:] for k,v in b.items()}
 for k in s:random.Random((seed<<8)+k).shuffle(s[k])
 n=sum(map(len,s.values()));p={k:len(v)/n for k,v in s.items()};u={k:0 for k in s};o=[]
 for k in range(4):o.append(s[k][0]);u[k]=1
 while len(o)<n:
  t=len(o)+1;avail=[k for k in range(4) if u[k]<len(s[k])];k=max(avail,key=lambda x:(t*p[x]-u[x],-x));o.append(s[k][u[k]]);u[k]+=1
 return o
def main():
 m=json.loads(M.read_text());r=json.loads(R.read_text());train=list(m['target_train']);val=set(m['target_validation']);test=set(m['target_evaluation']);b={k:[] for k in range(4)}
 for x in train:b[cl(float(r[x]))].append(x)
 if any(not b[k] for k in b):raise ValueError('missing training class')
 O.mkdir(parents=True,exist_ok=True)
 for seed in(25,42):
  o=order(b,seed);a={str(k):o[:v] for k,v in C.items()};cc={str(k):{str(c):sum(cl(float(r[x]))==c for x in ids) for c in range(4)} for k,ids in a.items()}
  assert all(set(ids)<=set(train) and not set(ids)&val and not set(ids)&test for ids in a.values()) and all(set(a[str(x)])<=set(a[str(y)]) for x,y in zip(C,tuple(C)[1:])) and all(all(z>0 for z in q.values()) for q in cc.values())
  d={'protocol':'m8_class_stratified_strict_nested_anchor_v1','city':'pittsburgh','training_seed':seed,'source_split_manifest_sha256':dg(M.read_bytes()),'risk_labels_file_sha256':dg(R.read_bytes()),'selection_rule':'target-train-label-only deterministic class-proportional interleaving; prefixes strict nested','anchor_counts':{str(k):v for k,v in C.items()},'anchor_ids':a,'anchor_class_counts':cc,'anchor_ids_sha256':{k:dg(''.join(f'{x}\n' for x in v).encode()) for k,v in a.items()},'nesting_checks':{f'{x}%_subset_{y}%':True for x,y in zip(C,tuple(C)[1:])},'integrity_checks':{'all_anchor_ids_in_target_train':True,'no_anchor_validation_overlap':True,'no_anchor_evaluation_overlap':True,'validation_or_test_labels_read':False}}
  (O/f'seed_{seed}.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
