"""Strict structural validation; no model calls or semantic relabeling."""
import collections
import hashlib
import json
import math
from pathlib import Path

NEEDS = {'current_sufficient','history_helpful','history_required','ambiguous'}
RELATIONS = {'direct_to_target','about_target','reply_dependency','shared_event','self_continuation','unrelated','uncertain'}
UTILITIES = {'decisive','helpful','weak','irrelevant','misleading','uncertain'}
DIRECTIONS = {'positive','negative','mixed','none','uncertain'}
TRANSITIONS = {'valence_flip','emotion_shift_same_valence','intensity_change','reinforcement','none','uncertain'}
BASES = {'explicit_reply','direct_address','discourse_dependency','semantic_context','none','uncertain'}
ANN_FIELDS = {'annotation_version','source_annotation_id','dialogue_id','target_turn_id','target_role','context_need','candidate_assessments','selected_evidence_turn_id','selection_reason','overall_confidence'}
CAND_FIELDS = {'candidate_turn_id','distance','candidate_speaker','role_relation','prediction_utility','impact_direction','transition_type','impact_strength','evidence_basis','confidence','short_reason'}
RUBRIC = {'causal_structural':20,'recent_window':15,'context_selection':15,'role_evidence':15,'prediction_utility':15,'impact_labels':15,'confidence_justification':5}


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records),encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def key(r):
    return str(r['dialogue_id']),int(r.get('target_turn_id',r.get('turn_id',-1)))


def unique(rows, name):
    out={}
    for r in rows:
        k=key(r)
        if k in out:raise ValueError(f'{name}: duplicate key {k}')
        out[k]=r
    return out


def number(x, lo, hi):
    return type(x) in (int,float) and math.isfinite(x) and lo<=x<=hi


def validate_annotation(a, src):
    problems=[]
    def check(condition, message):
        if not condition:problems.append(message)
    check(set(a)==ANN_FIELDS,'annotation schema')
    check(set(src)=={'annotation_id','dialogue_id','turn_id','history','target'},'blind source schema')
    check(set(src['target'])=={'turn_id','speaker','text'},'blind target schema')
    check(all(set(h)=={'turn_id','speaker','text'} for h in src['history']),'blind history schema')
    check(a.get('annotation_version')=='S6C-V2.3-local-impact','annotation version')
    check(key(a)==key(src),'source key')
    check(type(a.get('target_turn_id')) is int and a.get('target_turn_id')==src['turn_id'],'integer target ID')
    check(a.get('source_annotation_id')==src['annotation_id'],'source annotation ID')
    check(a.get('target_role')==src['target']['speaker'],'target speaker')
    check(a.get('context_need') in NEEDS,'context_need')
    check(number(a.get('overall_confidence'),0,1),'overall confidence')
    check(isinstance(a.get('selection_reason'),str) and len(a['selection_reason'].split())<=30,'selection reason')
    t=src['turn_id']
    check(type(t) is int and t>=0,'source target index')
    check(src['target']['turn_id']==t,'source target index alignment')
    check([h['turn_id'] for h in src['history']]==list(range(max(0,t-3),t)),'source causal history')
    hs={h['turn_id']:h for h in src['history']}
    cs=a.get('candidate_assessments')
    if not isinstance(cs,list):return problems+['candidate list']
    check([c.get('candidate_turn_id') for c in cs]==list(reversed(list(hs))),'candidate coverage/order')
    byid={}
    for c in cs:
        check(set(c)==CAND_FIELDS,'candidate schema')
        i=c.get('candidate_turn_id');byid[i]=c
        check(type(i)is int and i in hs,'candidate causal ID')
        if i in hs:
            check(c.get('distance')==t-i and type(c.get('distance'))is int,'distance')
            check(c.get('candidate_speaker')==hs[i]['speaker'],'candidate speaker')
        for field, allowed in [('role_relation',RELATIONS),('prediction_utility',UTILITIES),('impact_direction',DIRECTIONS),('transition_type',TRANSITIONS),('evidence_basis',BASES)]:
            check(c.get(field) in allowed,field)
        strength=c.get('impact_strength')
        check(strength is None or type(strength)is int and strength in (0,1,2,3),'impact strength')
        check(number(c.get('confidence'),0,1),'candidate confidence')
        check(isinstance(c.get('short_reason'),str) and len(c['short_reason'].split())<=30,'candidate reason')
        if c.get('impact_direction')=='none':
            check(c.get('transition_type')=='none' and strength==0,'none impact consistency')
        if c.get('role_relation')=='unrelated':
            check(c.get('prediction_utility') in {'irrelevant','misleading','uncertain'},'unrelated utility')
        if c.get('prediction_utility')=='decisive':check(number(c.get('confidence'),.7,1),'decisive confidence')
    selected=a.get('selected_evidence_turn_id')
    if selected is not None:
        check(type(selected)is int and selected in byid,'selected ID')
        c=byid.get(selected,{})
        check(c.get('prediction_utility') in {'helpful','decisive'} and c.get('role_relation')!='unrelated','selected utility/relation')
    if a.get('context_need') in {'current_sufficient','ambiguous'}:check(selected is None,'unneeded/ambiguous selection')
    if a.get('context_need') in {'history_helpful','history_required'}:check(selected is not None,'needed evidence absent')
    return problems


def validate_audit(a, annotation):
    errors=[]
    if key(a)!=key(annotation):errors.append('audit key')
    if a.get('audit_version')!='S6C-V2.3-self-audit-1.0':errors.append('audit version')
    if a.get('final_annotation')!=annotation:errors.append('final annotation mismatch')
    for prefix in ['original','final']:
        subs=a.get(prefix+'_subscores',{})
        if set(subs)!=set(RUBRIC) or any(not number(subs.get(k),0,m) for k,m in RUBRIC.items()):
            errors.append(prefix+' rubric bounds')
        elif sum(subs.values())!=a.get(prefix+'_total_score'):errors.append(prefix+' score arithmetic')
    action=a.get('action')
    if action not in {'kept','reannotated'}:errors.append('not accepted: '+str(action))
    if a.get('final_hard_failures')!=[] or not number(a.get('final_total_score'),90,100):errors.append('final audit failed')
    if action=='kept' and (a.get('reannotation_attempted') is not False or a.get('original_hard_failures')!=[] or not number(a.get('original_total_score'),90,100)):
        errors.append('kept action inconsistency')
    if action=='reannotated' and (a.get('reannotation_attempted') is not True or (number(a.get('original_total_score'),90,100) and not a.get('original_hard_failures'))):
        errors.append('reannotation action inconsistency')
    return errors


def check_bundle(folder):
    folder=Path(folder)
    inputs=read_jsonl(folder/'source_inputs.jsonl')
    annotations=read_jsonl(folder/'teacher_final.jsonl')
    audits=read_jsonl(folder/'teacher_audits.jsonl')
    im,am,um=[unique(rows,name) for rows,name in [(inputs,'source'),(annotations,'teacher'),(audits,'audit')]]
    if len(im)!=1500 or set(im)!=set(am) or set(im)!=set(um):
        raise ValueError('Expected 1500 unique matching source/teacher/audit keys')
    issues=[]
    for k,a in am.items():
        for error in validate_annotation(a,im[k])+validate_audit(um[k],a):
            issues.append({'key':list(k),'issue':error})
    if issues:raise ValueError(json.dumps({'annotation_validation_failed':len(issues),'examples':issues[:20]},ensure_ascii=False))
    return {'status':'passed_structural_checks','targets':len(im),'dialogues':len({k[0] for k in im}),
            'context_need':dict(collections.Counter(x['context_need'] for x in annotations)),
            'audit_actions':dict(collections.Counter(x['action'] for x in audits)),
            'independent_semantic_verification':False,
            'files_sha256':{n:sha(folder/n) for n in ['source_inputs.jsonl','teacher_final.jsonl','teacher_audits.jsonl']}}
