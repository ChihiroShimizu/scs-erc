#!/usr/bin/env python3
"""Use the server's existing Stage7 implementation in an isolated experiment."""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import zipfile
from data_checks import check_bundle,read_jsonl,write_json,write_jsonl,sha,key

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
CONFIG=PACKAGE/'config.yaml'


def balanced_label_control(records):
    """Two classification records + one identical replay per teacher target."""
    out=[]
    for record in records:
        out.append(record)
        if record.get('meta',{}).get('task')=='generic_emotion':
            duplicate=copy.deepcopy(record);duplicate['meta']['control_classification_repeat']=True
            out.append(duplicate)
    return out


def source_matches(target,source):
    if source['target']!={'turn_id':target.turn_index,'speaker':target.speaker,'text':target.text}:
        return False
    expected=[{'turn_id':target.turn_index-len(target.history)+i,'speaker':u.speaker,'text':u.text}
              for i,u in enumerate(target.history)]
    return source['history']==expected


def load_core():
    sys.path.insert(0,str(ROOT/'src'))
    try:
        core=importlib.import_module('pfa_erc.training.llm_evidence_supervision_stage7')
    except ImportError as exc:
        raise RuntimeError('Activate .venv-server and confirm the existing Stage7 module is installed in src/. '+str(exc)) from exc
    required=['preflight','prepare_training_data','train_variant','evaluate_seed','LLMEngine',
        'SFTDataset','_plan_from_annotation','_split_dialogues','build_emotion_prompt','_save_score_records','_metrics']
    missing=[n for n in required if not hasattr(core,n)]
    if missing:raise RuntimeError('Unsupported server Stage7 API: '+str(missing))
    old_version=core.VERSION
    if old_version!='2.0.0':
        raise RuntimeError(f'Expected Stage7 v2.0.0 API, found {old_version}; source file: {core.__file__}')
    section=core._section(core._read_yaml(CONFIG))
    core._project_root=lambda config_path:ROOT
    core.VERSION='7R-1.0.0';core.EXPERIMENT='llm_evidence_supervision_meld_stage7r1500'
    original={'preflight':core.preflight,'prepare':core.prepare_training_data,'train':core.train_variant,
              'evaluate':core.evaluate_seed,'engine':core.LLMEngine,'dataset':core.SFTDataset}

    class NoTruncateDataset(original['dataset']):
        def __getitem__(self,index):
            r=self.records[index]
            length=len(self._render_prompt(r['messages']))+len(self.tokenizer.encode(str(r['completion']),add_special_tokens=False))
            length+=int(self.tokenizer.eos_token_id is not None)
            if length>self.max_length:raise ValueError(f'Truncation forbidden: SFT {r.get("meta",{}).get("key")} length={length}, limit={self.max_length}')
            return super().__getitem__(index)

    class NoTruncateEngine(original['engine']):
        def _render(self,system,prompt):
            rendered=super()._render(system,prompt)
            length=len(self.tokenizer.encode(rendered,add_special_tokens=False))
            reserve=max(len(x) for x in self.label_ids)
            if length+reserve>self.max_input_tokens:
                raise ValueError(f'Truncation forbidden: inference length={length}, reserve={reserve}, limit={self.max_input_tokens}')
            return rendered
    core.SFTDataset=NoTruncateDataset;core.LLMEngine=NoTruncateEngine
    return core,original,section


def state_signature(core,section):
    files={'config':CONFIG,'implementation':Path(core.__file__),
           'wrapper':Path(__file__),'statistics':PACKAGE/'statistics_report.py',
           'data_checks':PACKAGE/'data_checks.py',
           'train':ROOT/section['processed_dir']/'train.jsonl',
           'validation':ROOT/section['processed_dir']/'validation.jsonl',
           **{n:PACKAGE/'data'/n for n in ['teacher_final.jsonl','teacher_audits.jsonl','source_inputs.jsonl']}}
    hashes={n:sha(p) for n,p in files.items()}
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return signature,hashes


def freeze(core,section,run_name):
    paths=core._paths(ROOT,section,run_name)
    signature,hashes=state_signature(core,section)
    signature=hashlib.sha256((signature+':'+run_name).encode()).hexdigest()
    path=paths['base']/'frozen_inputs.json'
    if path.exists():
        if json.loads(path.read_text())['run_signature']!=signature:
            raise RuntimeError('Inputs/code/config changed since this run was frozen; use a new artifact directory, do not mix caches')
    else:
        if any(paths['base'].glob('adapters/seed*/*/training_complete.json')):
            raise RuntimeError('Found untracked cached adapters in the new experiment directory')
        write_json(path,{'run_signature':signature,'run_name':run_name,'files_sha256':hashes,
            'server_core':str(Path(core.__file__).resolve()),'test_locked':True})
    return paths,signature


def preflight(core,original,section,run_name='validation'):
    checked=check_bundle(PACKAGE/'data')
    sources={key(r):r for r in read_jsonl(PACKAGE/'data/source_inputs.jsonl')}
    train=core.read_jsonl(ROOT/section['processed_dir']/'train.jsonl')
    targets=core._target_map(train,max_history=3)
    core_ids,cal_ids=core._split_dialogues([d.dialogue_id for d in train],.10,2026)
    counts=collections.Counter()
    for k,src in sources.items():
        target=targets.get(f'{k[0]}::{k[1]}')
        if target is None or not source_matches(target,src):
            raise ValueError(f'Annotation input does not match this server MELD train version: {k}')
        counts['calibration' if k[0] in cal_ids else 'train']+=1
    if counts!={'train':1200,'calibration':300}:
        raise ValueError(f'Expected 1200/300 using original dialogue split; got {dict(counts)}. Check the original preparation manifest.')
    result=original['preflight'](CONFIG,run_name)
    result.update({'teacher_source':'new 1500 user-supplied model annotations with internal self-audit; exact model provenance kept in data/source_provenance.json',
        'teacher_train_records':counts['train'],'teacher_calibration_records':counts['calibration'],
        'independent_verification':False,'annotation_checks':checked,
        'context_gate_experiment':False,'free_text_CoT_training':False})
    paths,signature=freeze(core,section,run_name)
    result['run_signature']=signature
    write_json(paths['preflight'],result)
    write_json(paths['base']/'dialogue_split.json',{'train':sorted(core_ids),'calibration':sorted(cal_ids),'split_seed':2026})
    print(f'Stage7R preflight passed: 1500 annotations, 1200 SFT / 300 calibration, signature={signature[:12]}',flush=True)
    return result


def token_audit(core,section,paths,signature):
    report_path=paths['reports']/'token_length_audit.json'
    if report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous.get('run_signature')==signature:
            if previous.get('overlong_records'):raise ValueError('Previously failed token audit; training remains blocked')
            return
    tok=core._load_tokenizer(section);limit=int(section['max_sequence_tokens']);report={};over=[]
    for variant in core.TRAIN_VARIANTS:
        records=read_jsonl(paths['data']/f'{variant}_train.jsonl')
        dataset=core.SFTDataset(records,tok,limit,enable_thinking=False)
        maximum=0;completion_tokens=0
        for i,r in enumerate(records):
            prompt=dataset._render_prompt(r['messages'])
            completion=tok.encode(str(r['completion']),add_special_tokens=False)
            n=len(prompt)+len(completion)+int(tok.eos_token_id is not None)
            maximum=max(maximum,n);completion_tokens+=len(completion)+int(tok.eos_token_id is not None)
            if n>limit:over.append({'variant':variant,'key':r.get('meta',{}).get('key'),'tokens':n})
            if i and i%2000==0:print(f'[token audit] {variant} {i}/{len(records)}',flush=True)
        report[variant]={'records':len(records),'max_sequence_tokens_observed':maximum,
            'supervised_tokens_in_dataset':completion_tokens,
            'task_counts':dict(collections.Counter(r.get('meta',{}).get('task') for r in records))}
    write_json(report_path,{'run_signature':signature,'configured_limit':limit,'variants':report,
        'overlong_records':over,'truncation_permitted':False,'equal_FLOPs_claimed':False})
    if over:raise ValueError(f'{len(over)} records exceed shared token limit; inspect token_length_audit.json and freeze a new configuration before training')


def prepare(core,original,section,run_name='validation'):
    preflight(core,original,section,run_name)
    paths,signature=freeze(core,section,run_name)
    marker=paths['data']/'s7r_prepared.json'
    if marker.exists():
        stored=json.loads(marker.read_text())
        if stored['run_signature']!=signature:raise RuntimeError('Prepared signature mismatch')
        for name,digest in stored['prepared_sha256'].items():
            if sha(paths['data']/name)!=digest:raise RuntimeError('Prepared training data changed: '+name)
        token_audit(core,section,paths,signature)
        return stored
    audit=original['prepare'](CONFIG,run_name)
    if (audit['teacher_train_records'],audit['teacher_calibration_records'])!=(1200,300):raise ValueError('Teacher split count mismatch')
    generic_path=paths['data']/'generic_reasoning_train.jsonl'
    generic=balanced_label_control(read_jsonl(generic_path))
    write_jsonl(generic_path,generic)
    for variant in core.TRAIN_VARIANTS[1:]:
        records=read_jsonl(paths['data']/f'{variant}_train.jsonl')
        if len(records)!=3600:raise ValueError(f'Expected 3600 records for {variant}, got {len(records)}')
    # All replay/base records and teacher targets must be from train-core.
    split=json.loads((paths['base']/'dialogue_split.json').read_text());cal=set(split['calibration'])
    for variant in core.TRAIN_VARIANTS:
        for r in read_jsonl(paths['data']/f'{variant}_train.jsonl'):
            meta=r['meta'];d=meta.get('dialogue_id') or str(meta['key']).rsplit('::',1)[0]
            if d in cal:raise RuntimeError('Calibration dialogue leaked into training')
    audit.update({'teacher_train_records':1200,'teacher_calibration_records':300,
        'branch_examples':{v:len(read_jsonl(paths['data']/f'{v}_train.jsonl')) for v in core.TRAIN_VARIANTS},
        'label_only_control':'two identical emotion classifications plus one shared replay per teacher target',
        'equal_supervised_tokens_or_FLOPs_claimed':False,'run_signature':signature})
    target_map=core._target_map(core.read_jsonl(ROOT/section['processed_dir']/'train.jsonl'),max_history=3)
    for partition in ['train','calibration']:
        anns=read_jsonl(paths['data']/f'teacher_{partition}_annotations.jsonl')
        audit[f'teacher_{partition}_emotion_counts']=dict(collections.Counter(
            target_map[f"{a['dialogue_id']}::{a['target_turn_id']}"].gold for a in anns))
        audit[f'teacher_{partition}_dialogues']=len({a['dialogue_id'] for a in anns})
    write_json(paths['data_audit'],audit)
    prepared={p.name:sha(p) for p in sorted(paths['data'].glob('*.jsonl'))}
    result={'run_signature':signature,'prepared_sha256':prepared,'data_audit':audit}
    write_json(marker,result)
    token_audit(core,section,paths,signature)
    print('Prepared all branches: 1200 teacher train, 300 teacher calibration, 3600 records per continued-training branch.',flush=True)
    return result


def weight_hash(directory):
    files=sorted(Path(directory).glob('adapter_model*'))+[Path(directory)/'adapter_config.json']
    return {f.name:sha(f) for f in files if f.is_file()}


def train(core,original,section,variant,seed,run_name='validation',smoke=False):
    paths,signature=freeze(core,section,run_name)
    output=core._adapter_dir(paths,seed,variant)
    guard=output/'s7r_training_identity.json'
    identity={'run_signature':signature,'variant':variant,'seed':seed,'smoke':smoke,
        'prepared_data_sha256':sha(paths['data']/f'{variant}_train.jsonl')}
    parent=core._adapter_dir(paths,seed,'emotion_base') if variant!='emotion_base' else None
    before=weight_hash(parent) if parent else None
    identity['parent_weights']=before
    if guard.exists() and json.loads(guard.read_text())!=identity:raise RuntimeError('Adapter training identity mismatch')
    if not guard.exists() and (output/'training_complete.json').exists():raise RuntimeError('Untracked completed adapter; refusing to reuse')
    if (output/'training_complete.json').exists() and (output/'s7r_verified.json').exists():
        verified=json.loads((output/'s7r_verified.json').read_text())
        if verified['run_signature']!=signature or verified['adapter_weights']!=weight_hash(output):
            raise RuntimeError('Completed adapter bytes changed; refusing cached training')
    write_json(guard,identity)
    result=original['train'](CONFIG,variant,seed,run_name,
        max_steps_override=2 if smoke else None,max_examples=16 if smoke else None)
    if parent and weight_hash(parent)!=before:raise RuntimeError('Parent adapter mutated during branch training')
    expected=2 if smoke else int(section['emotion_base_max_steps'] if variant=='emotion_base' else section['branch_max_steps'])
    if result['optimizer_steps']!=expected:raise RuntimeError('Training step count mismatch')
    write_json(output/'s7r_verified.json',{'run_signature':signature,'steps':expected,
        'adapter_weights':weight_hash(output),'parent_unchanged':True})


def controlled_prompt(core,target,mode,selected=None):
    prompt=core.build_emotion_prompt(target,core.LABELS,mode=mode,selected_turn_id=selected)
    return "Task: classify the target speaker's expressed emotion. Use only the supplied earlier evidence when relevant.\n"+prompt.split('\n',1)[1]


def mechanism_evaluation(core,section,paths,seed,max_targets=None):
    import numpy as np
    cal,dev=core._load_inference_targets(ROOT,section,paths)
    if max_targets is not None:cal,dev=cal[:max_targets],dev[:max_targets]
    engine=core.LLMEngine(section,core._adapter_dir(paths,seed,'full_supervised'))
    results={};goldcal=[core.LABELS.index(t.gold) for t in cal]
    for name,mode in [('mechanism_current','current'),('mechanism_recent3','recent3'),('mechanism_selected','selected')]:
        score_parts={}
        for split,targets in [('calibration',cal),('validation',dev)]:
            plan_rows=read_jsonl(paths['scores']/f'seed{seed}_full_supervised_{split}.jsonl')
            if [r['key'] for r in plan_rows]!=[t.key for t in targets]:raise ValueError('Cached plan/target alignment')
            prompts=[controlled_prompt(core,t,mode,r['plan']['selected_evidence_turn_id']) for t,r in zip(targets,plan_rows)]
            print(f'[mechanism] seed={seed} {name} {split}: {len(targets)} targets',flush=True)
            scores=engine.score_emotions(prompts);score_parts[split]=scores
            core._save_score_records(paths['scores']/f'seed{seed}_{name}_{split}.jsonl',targets,scores,None)
        temperature=core._fit_temperature(score_parts['calibration'],goldcal)
        m=core._metrics(score_parts['validation'],[t.gold for t in dev],core.LABELS,temperature)
        results[name]={k:v for k,v in m.items() if k!='raw_scores'}
    del engine;gc.collect();core.torch.cuda.empty_cache()
    return results


def evaluate(core,original,section,seed,run_name='validation',smoke=False):
    paths,signature=freeze(core,section,run_name)
    marker=paths['scores']/f'seed{seed}_s7r_complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['run_signature']!=signature:raise RuntimeError('Evaluation identity mismatch')
        print(f'Skip completed evaluation seed={seed}');return
    for v in core.TRAIN_VARIANTS:
        state=json.loads((core._adapter_dir(paths,seed,v)/'s7r_verified.json').read_text())
        if state['run_signature']!=signature:raise RuntimeError('Unverified training cache')
        if weight_hash(core._adapter_dir(paths,seed,v))!=state['adapter_weights']:raise RuntimeError('Adapter bytes changed after training')
    result=original['evaluate'](CONFIG,seed,run_name,max_targets=8 if smoke else None)
    result['variants'].update(mechanism_evaluation(core,section,paths,seed,max_targets=8 if smoke else None))
    write_json(paths['scores']/f'seed{seed}_validation.json',result)
    write_json(marker,{'run_signature':signature,'seed':seed,'targets':result['targets'],'smoke':smoke})


def controller_report(core,section,paths,seeds):
    from sklearn.metrics import precision_recall_fscore_support,f1_score
    teachers={f"{a['dialogue_id']}::{a['target_turn_id']}":a for a in read_jsonl(paths['data']/'teacher_calibration_annotations.jsonl')}
    results=[]
    for seed in seeds:
        for variant in ['evidence_supervised','full_supervised','shuffled_supervised']:
            rs=read_jsonl(paths['scores']/f'seed{seed}_{variant}_calibration.jsonl')
            matched=[(teachers[r['key']],r['plan']) for r in rs if r['key'] in teachers]
            if len(matched)!=300:raise ValueError('Missing held-out teacher diagnostics')
            y=[a['context_need'] in {'history_helpful','history_required'} for a,p in matched]
            pred=[p['context_need'] in {'history_helpful','history_required'} for a,p in matched]
            pr,re,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
            sufficient=[(a,p) for a,p in matched if a['context_need']=='current_sufficient']
            needed=[(a,p) for a,p in matched if a['context_need'] in {'history_helpful','history_required'}]
            results.append({'seed':seed,'variant':variant,'records':300,
                'context_need_macro_f1':float(f1_score([a['context_need'] for a,p in matched],[p['context_need'] for a,p in matched],
                    labels=['current_sufficient','history_helpful','history_required','ambiguous'],average='macro',zero_division=0)),
                'history_need_precision':float(pr),'history_need_recall':float(re),'history_need_f1':float(f),
                'current_sufficient_support':len(sufficient),
                'unnecessary_history_selection_rate':sum(p['selected_evidence_turn_id'] is not None for a,p in sufficient)/len(sufficient) if sufficient else None,
                'needed_support':len(needed),
                'needed_evidence_exact_accuracy':sum(p['selected_evidence_turn_id']==a['selected_evidence_turn_id'] for a,p in needed)/len(needed) if needed else None,
                'parse_success_rate':sum(bool(p['parse_success']) for a,p in matched)/len(matched),
                'teacher_agreement_not_human_gold':True})
    write_json(paths['reports']/'controller_diagnostics.json',results)


def status(core,section):
    paths=core._paths(ROOT,section,'validation')
    result={'artifact_dir':str(paths['base']),'seeds':{},'test_locked':True}
    for seed in section['seeds']:
        result['seeds'][seed]={'adapters':{v:(core._adapter_dir(paths,seed,v)/'training_complete.json').exists() for v in core.TRAIN_VARIANTS},
            'evaluation':(paths['scores']/f'seed{seed}_s7r_complete.json').exists()}
    result['report']=(paths['reports']/'validation_summary.md').exists()
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['test','preflight','prepare','smoke','run','train-seed','evaluate-seed','report','status','export'])
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.command=='test':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(PACKAGE/'tests'))
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    core,original,section=load_core()
    # Existing module functions call their process-local wrappers; server file stays unchanged.
    core.preflight=lambda config_path,run_name='validation':preflight(core,original,section,run_name)
    core.prepare_training_data=lambda config_path,run_name='validation':prepare(core,original,section,run_name)
    if args.command=='status':status(core,section);return
    paths=core._paths(ROOT,section,'validation');paths['base'].mkdir(parents=True,exist_ok=True)
    with (paths['base']/'.execution.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another Stage7R command is already running in this artifact directory')
        if args.command=='preflight':preflight(core,original,section)
        elif args.command=='prepare':prepare(core,original,section)
        elif args.command=='smoke':
            run_name=f'smoke_seed{args.seed}'
            prepare(core,original,section,run_name)
            for v in core.TRAIN_VARIANTS:train(core,original,section,v,args.seed,run_name,smoke=True)
            evaluate(core,original,section,args.seed,run_name,smoke=True)
            print('SMOKE PASSED. Smoke artifacts are isolated from the main run.')
        elif args.command in {'run','train-seed','evaluate-seed'}:
            prepare(core,original,section)
            if args.command=='run':
                for seed in section['seeds']:
                    for v in core.TRAIN_VARIANTS:train(core,original,section,v,seed)
                for seed in section['seeds']:evaluate(core,original,section,seed)
            elif args.command=='train-seed':
                if args.seed not in section['seeds']:raise ValueError('Unregistered seed')
                for v in core.TRAIN_VARIANTS:train(core,original,section,v,args.seed)
            else:
                if args.seed not in section['seeds']:raise ValueError('Unregistered seed')
                evaluate(core,original,section,args.seed)
            if args.command=='run':
                from statistics_report import report
                _,signature=freeze(core,section,'validation')
                controller_report(core,section,paths,section['seeds'])
                report(paths['base'],section['seeds'],section,signature)
                print('ALL REGISTERED EXPERIMENTS COMPLETE. Test remains locked.')
        elif args.command=='report':
            from statistics_report import report
            _,signature=freeze(core,section,'validation')
            controller_report(core,section,paths,section['seeds'])
            report(paths['base'],section['seeds'],section,signature)
        elif args.command=='export':
            if not (paths['reports']/'validation_summary.md').exists():raise RuntimeError('Final report is not complete; use status to inspect progress')
            target=paths['base']/'MELD_Stage7R_1500_results.zip'
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
                for sub in ['reports','scores']:
                    for f in sorted((paths['base']/sub).glob('*')):
                        if f.is_file():z.write(f,str(f.relative_to(paths['base'])))
                for f in [paths['base']/'frozen_inputs.json',paths['base']/'dialogue_split.json',paths['data_audit'],CONFIG]:
                    z.write(f,f.name)
            print(target)


if __name__=='__main__':main()
