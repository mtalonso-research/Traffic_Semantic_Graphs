"""Persistent M8 screen: fixed 29 target updates/epoch across class-stratified nested anchors."""
from pathlib import Path
import json, subprocess, sys
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiment_results/adaptation_repair'; ANCHORS=OUT/'m7_stratified_boston_anchors'; LOG=OUT/'m8_fixed_exposure_controller.log'
def note(x):
 line=f'{datetime.now().isoformat()} {x}';print(line,flush=True);LOG.open('a',encoding='utf-8').write(line+'\n')
def valid(s,r):
 p=OUT/f'm8_boston_seed{s}_a{r}/result.json'
 if not p.exists():return False
 d=json.loads(p.read_text());return d.get('protocol')=='m8_fixed_target_exposure_validation_f1_v1' and d.get('test_accessed') is False and d.get('seed')==s and d.get('anchor_pct')==r
def one(s,r):
 if valid(s,r):note(f'reuse m8_boston_seed{s}_a{r}');return
 out=OUT/f'm8_boston_seed{s}_a{r}'
 if (out/'result.json').exists():raise RuntimeError(f'invalid existing M8 result {out}')
 out.mkdir(parents=True,exist_ok=True);f='experiment_results/nested_scene_split/frozen_boston'
 cmd=[sys.executable,'-u','experiment_results/anchor_multiseed/run_anchor_experiment.py','--target-city','boston','--city-view-root',f'{f}/city_views/singapore_to_boston','--ae-clean','experiment_results/nested_scene_split/pilots/boston_seed42/autoencoders/singapore_ae_best_model.pt','--ae-noisy','experiment_results/nested_scene_split/pilots/boston_seed42/autoencoders/boston_ae_best_model.pt','--manifest-root',f'{f}/manifests/singapore_to_boston','--anchors-file',str(ANCHORS/f'seed_{s}.json'),'--anchor-pct',str(r),'--seed',str(s),'--risk-checkpoint',f'experiment_results/nested_scene_split/healthy_source_classifier_weighted_seed_{s}.pt','--lambda-anchor','1.0','--epochs','150','--patience','15','--paired-anchor-batches','--full-anchor-epochs','--fixed-target-updates-per-epoch','29','--anchor-weighted-ce','--selection-metric','macro_f1','--m6-train-risk','--m8-fixed-exposure','--risk-learning-rate','1e-5','--source-replay-steps','2','--output-dir',str(out),'--device','cuda:0']
 note(f'launch {out.name}; fixed 29 target updates/epoch, complete anchor pass then deterministic cycle, validation-only')
 with (out/'train.log').open('a') as so,(out/'train.err.log').open('a') as se:code=subprocess.run(cmd,cwd=ROOT,stdout=so,stderr=se).returncode
 if code or not valid(s,r):raise RuntimeError(f'M8 failed {out}')
 note(f'completed {out.name}')
def main():
 note('M8 begins after unequal M6/M7 target exposure was diagnosed; no target test access')
 for s in(25,42):
  for r in(3,5,10,20,50):one(s,r)
 note('M8 full Boston curve complete')
if __name__=='__main__':main()
