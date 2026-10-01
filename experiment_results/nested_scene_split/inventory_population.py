"""Inventory canonical city graphs by metadata scene and database without mutating data."""
from __future__ import annotations
import argparse, json, hashlib
from collections import Counter, defaultdict
from pathlib import Path

def cls(x): return 3 if x>.3442 else 2 if x>.1008 else 1 if x>.0043 else 0
def sha(xs): return hashlib.sha256(''.join(f'{x}\n' for x in xs).encode()).hexdigest()

def main(a):
    graphs=Path(a.graph_root); risks=json.loads(Path(a.risks).read_text())
    cities=defaultdict(list); bad=[]
    for path in graphs.glob('*_graph.json'):
        eid=path.name.split('_')[0]
        try: data=json.loads(path.read_text(encoding='utf-8')); m=data.get('metadata',{}); city=m.get('city','UNKNOWN'); scene=m.get('scene_token') or m.get('scene_name'); db=m.get('db_file')
        except Exception as e: bad.append({'sample_id':eid,'error':str(e)}); continue
        if eid not in risks or not scene: bad.append({'sample_id':eid,'error':'missing risk or scene'}); continue
        cities[city].append({'sample_id':eid,'scene_id':scene,'db_file':db,'label':cls(risks[eid]),'frames':m.get('frames'),'path':str(path)})
    out={'protocol':'scene_aware_nested_anchor_v1','grouping_key':'metadata.scene_token inventory; database/log grouping will be used for the frozen split','grouping_rationale':'Each scene token occurs exactly once in this corpus, so scene grouping alone equals graph-level splitting. Database ID is a stricter available boundary and will prevent cross-split log leakage.','corrupt_or_unusable':bad,'cities':{}}
    for city, rows in sorted(cities.items()):
        byscene=defaultdict(list); bydb=defaultdict(list)
        for r in rows: byscene[r['scene_id']].append(r); bydb[r['db_file']].append(r)
        out['cities'][city]={'sample_count':len(rows),'scene_count':len(byscene),'database_count':len(bydb),'label_counts':dict(sorted(Counter(str(r['label']) for r in rows).items())),'frame_counts':dict(sorted(Counter(str(r['frames']) for r in rows).items())),'samples_per_scene':{'min':min(map(len,byscene.values())),'max':max(map(len,byscene.values())),'mean':len(rows)/len(byscene)},'scene_records':[{'scene_id':s,'database_id':next((r['db_file'] for r in rs),None),'sample_ids':sorted(r['sample_id'] for r in rs),'label_counts':dict(sorted(Counter(str(r['label']) for r in rs).items()))} for s,rs in sorted(byscene.items())],'sample_ids_sha256':sha(sorted(r['sample_id'] for r in rows))}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({c:{k:v[k] for k in ('sample_count','scene_count','database_count','label_counts','frame_counts','samples_per_scene')} for c,v in out['cities'].items()},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--graph-root',required=True);p.add_argument('--risks',required=True);p.add_argument('--output',default='experiment_results/nested_scene_split/inventory.json');main(p.parse_args())
