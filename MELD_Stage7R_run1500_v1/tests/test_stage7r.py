import ast
import collections
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from data_checks import check_bundle,read_jsonl,validate_annotation,validate_audit,write_json,write_jsonl
from run_stage7r import balanced_label_control,source_matches,controlled_prompt


def reference_functions():
    """Exercise actual server preparation functions without importing CUDA code."""
    import numpy as np
    import yaml
    source=Path(os.environ.get('STAGE7_REFERENCE',str(PACKAGE.parent/'src/pfa_erc/training/llm_evidence_supervision_stage7.py')))
    if not source.is_file():raise FileNotFoundError('Missing server Stage7 source: '+str(source))
    tree=ast.parse(source.read_text())
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for n in tree.body:
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)) or isinstance(n,ast.ClassDef) and n.name=='Target':
            nodes.append(n)
    module=types.ModuleType('_stage7_pure_fixture');sys.modules[module.__name__]=module
    module.__dict__.update({'np':np,'yaml':yaml,'json':json,'Path':Path,'hashlib':hashlib,
        'random':random,'defaultdict':collections.defaultdict,'Counter':collections.Counter,
        'dataclass':dataclasses.dataclass,'__file__':str(source)})
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),module.__dict__)
    return module


class AnnotationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=read_jsonl(PACKAGE/'data/source_inputs.jsonl')
        cls.annotations=read_jsonl(PACKAGE/'data/teacher_final.jsonl')
        cls.audits=read_jsonl(PACKAGE/'data/teacher_audits.jsonl')

    def test_complete_1500_unique_inputs_and_accepted_audits(self):
        r=check_bundle(PACKAGE/'data')
        self.assertEqual(r['targets'],1500)
        self.assertEqual(sum(r['audit_actions'].values()),1500)

    def test_future_evidence_and_missing_candidates_rejected(self):
        i=next(i for i,s in enumerate(self.sources) if s['history'])
        a=copy.deepcopy(self.annotations[i]);a['candidate_assessments'][0]['candidate_turn_id']=self.sources[i]['turn_id']
        self.assertTrue(validate_annotation(a,self.sources[i]))
        a=copy.deepcopy(self.annotations[i]);a['candidate_assessments'].pop()
        self.assertTrue(validate_annotation(a,self.sources[i]))

    def test_failed_or_inconsistent_audit_rejected(self):
        a=copy.deepcopy(self.audits[0]);a['final_total_score']=89
        self.assertTrue(validate_audit(a,self.annotations[0]))
        a=copy.deepcopy(self.audits[0]);a['final_annotation']['target_role']='tampered'
        self.assertTrue(validate_audit(a,self.annotations[0]))


class PreparationChecks(unittest.TestCase):
    def test_actual_legacy_preparation_balanced_and_dialogue_disjoint(self):
        import yaml
        core=reference_functions()
        dialogues=[];annotations=[]
        for d in range(8):
            turns=[types.SimpleNamespace(utterance_id=f'{d}:{t}',speaker=f'person{t%2}',text=f'turn {t}',emotion='neutral') for t in range(3)]
            dialogue=types.SimpleNamespace(dialogue_id=f'train:{d}',utterances=turns);dialogues.append(dialogue)
            for t,u in enumerate(turns):
                annotations.append({'annotation_version':'S6C-V2.3-local-impact','source_annotation_id':f'{d}:{t}',
                    'dialogue_id':dialogue.dialogue_id,'target_turn_id':t,'target_role':u.speaker,
                    'context_need':'current_sufficient','candidate_assessments':[],
                    'selected_evidence_turn_id':None,'selection_reason':'Current statement is clear.','overall_confidence':.8})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'processed').mkdir()
            (root/'processed/train.jsonl').write_text('')
            write_jsonl(root/'teacher.jsonl',annotations)
            config=root/'config.yaml';config.write_text(yaml.safe_dump({'llm_evidence_supervision_stage7':{
                'processed_dir':'processed','teacher_annotation_file':'teacher.jsonl',
                'artifact_dir':'out','split_seed':2026,'calibration_dialogue_fraction':.25,'emotion_replay_per_teacher':1}}))
            core.read_jsonl=lambda p:dialogues
            core.preflight=lambda *a,**k:None
            core._project_root=lambda p:root
            core.prepare_training_data(config)
            prepared=root/'out/prepared'
            generic=read_jsonl(prepared/'generic_reasoning_train.jsonl')
            nteacher=len(read_jsonl(prepared/'teacher_train_annotations.jsonl'))
            balanced=balanced_label_control(generic)
            self.assertEqual(len(balanced),3*nteacher)
            self.assertEqual(sum(r['meta']['task']=='generic_emotion' for r in balanced),2*nteacher)
            full=read_jsonl(prepared/'full_supervised_train.jsonl')
            self.assertEqual(len(full),len(balanced))
            self.assertEqual([r for r in balanced if r['meta']['task']=='emotion_current'],
                             [r for r in full if r['meta']['task']=='emotion_current'])
            cals={a['dialogue_id'] for a in read_jsonl(prepared/'teacher_calibration_annotations.jsonl')}
            for variant in core.TRAIN_VARIANTS:
                for r in read_jsonl(prepared/f'{variant}_train.jsonl'):
                    meta=r['meta'];d=meta.get('dialogue_id') or meta['key'].rsplit('::',1)[0]
                    self.assertNotIn(d,cals)

    def test_source_text_match_and_common_mechanism_prompt(self):
        core=reference_functions()
        u=types.SimpleNamespace(utterance_id='u0',speaker='a',text='earlier',emotion='neutral')
        t=core.Target('train:1',1,'u1','b','current','neutral',(u,))
        src={'target':{'turn_id':1,'speaker':'b','text':'current'},'history':[{'turn_id':0,'speaker':'a','text':'earlier'}]}
        self.assertTrue(source_matches(t,src));src['history'][0]['text']='changed'
        self.assertFalse(source_matches(t,src))
        current=controlled_prompt(core,t,'current')
        recent=controlled_prompt(core,t,'recent3')
        selected=controlled_prompt(core,t,'selected',0)
        self.assertEqual(recent,selected)
        self.assertEqual(current.splitlines()[0],selected.splitlines()[0])
        self.assertNotIn('Controller decision:',selected)


class StatisticsChecks(unittest.TestCase):
    def test_identical_methods_have_identical_paired_draws(self):
        import numpy as np
        from statistics_report import paired_bootstrap
        cm=np.zeros((2,2,3,7,7),dtype=int)
        cm[:,:,0,0,0]=2;cm[:,:,1,1,2]=3;cm[:,:,2,2,2]=4
        values=paired_bootstrap(cm,100,17)
        np.testing.assert_array_equal(values[:,0],values[:,1])

    def test_full_report_recomputes_f1_and_checks_target_alignment(self):
        import numpy as np
        from statistics_report import report,METHODS,LABELS,cm_metrics
        seeds=[42,43];signature='fixture-signature'
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);(base/'scores').mkdir()
            for seed in seeds:
                variants={}
                for mi,method in enumerate(METHODS):
                    rows=[];probs=[];preds=[];cm=np.zeros((7,7),dtype=int)
                    for i in range(28):
                        y=i%7;p=(y+int((i+mi+seed)%5==0))%7
                        scores=[-1.0]*7;scores[p]=2.0
                        prob=np.exp(scores);prob/=prob.sum();probs.append(prob.tolist());preds.append(LABELS[p]);cm[y,p]+=1
                        rows.append({'key':f'validation:{i//4}::{i%4}','dialogue_id':f'validation:{i//4}',
                            'gold':LABELS[y],'scores':scores})
                    wf,mf,acc,_=cm_metrics(cm)
                    variants[method]={'weighted_f1':float(wf),'macro_f1':float(mf),'accuracy':float(acc),
                        'nll':1.0,'brier':.5,'ece':.1,'temperature':1.0,'predictions':preds,'probabilities':probs}
                    if method!='entropy_router':write_jsonl(base/'scores'/f'seed{seed}_{method}_validation.jsonl',rows)
                write_json(base/'scores'/f'seed{seed}_validation.json',{'labels':LABELS,'targets':28,'variants':variants})
                write_json(base/'scores'/f'seed{seed}_s7r_complete.json',{'run_signature':signature})
            result=report(base,seeds,{'bootstrap_samples':40,'permutation_samples':50},signature)
            self.assertFalse(result['comparisons']['full_supervised_minus_generic_reasoning']['ensemble_used'])
            self.assertEqual(result['dialogues'],7)
            path=base/'scores/seed42_full_supervised_validation.jsonl'
            rs=read_jsonl(path);rs[0]['key']='mismatched';write_jsonl(path,rs)
            with self.assertRaises(ValueError):report(base,seeds,{'bootstrap_samples':10,'permutation_samples':10},signature)


if __name__=='__main__':unittest.main(verbosity=2)
