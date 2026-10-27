"""Dialogue-paired statistics on mean seed performance, not an ensemble."""
import csv
import collections
import json
from pathlib import Path
import numpy as np
from data_checks import read_jsonl, write_json

LABELS=['anger','disgust','fear','joy','neutral','sadness','surprise']
METHODS=['current_only','always_recent3','entropy_router','generic_reasoning',
         'evidence_supervised','full_no_impact','full_supervised','shuffled_supervised',
         'mechanism_current','mechanism_recent3','mechanism_selected']
NAMES={'current_only':'基础模型：仅当前句','always_recent3':'基础模型：前三轮',
       'entropy_router':'基础模型：不确定时参考历史','generic_reasoning':'仅情绪标签继续训练',
       'evidence_supervised':'仅需求与证据监督','full_no_impact':'完整训练，分类时只读证据',
       'full_supervised':'完整额外监督（SCS）','shuffled_supervised':'打乱额外标签',
       'mechanism_current':'同一完整adapter：当前句','mechanism_recent3':'同一完整adapter：前三轮',
       'mechanism_selected':'同一完整adapter：所选证据'}


def cm_metrics(cm):
    support=cm.sum(axis=-1)
    predicted=cm.sum(axis=-2)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=support+predicted
    f=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
    n=support.sum(axis=-1)
    wf=np.divide((support*f).sum(axis=-1),n,out=np.zeros_like(n,dtype=float),where=n>0)
    acc=np.divide(tp.sum(axis=-1),n,out=np.zeros_like(n,dtype=float),where=n>0)
    return wf,f.mean(axis=-1),acc,f


def confusion(y,pred,dialogue_ids):
    ds=sorted(set(dialogue_ids));dm={d:i for i,d in enumerate(ds)}
    out=np.zeros((len(ds),7,7),dtype=np.int64)
    for i,(a,b) in enumerate(zip(y,pred)):out[dm[dialogue_ids[i]],a,b]+=1
    return out,ds


def paired_bootstrap(cms,samples=10000,seed=20261021):
    # cms: methods x seeds x dialogues x gold x predicted
    rng=np.random.default_rng(seed);d=cms.shape[2];results=[]
    for start in range(0,samples,250):
        weights=rng.multinomial(d,np.full(d,1/d),size=min(250,samples-start))
        drawn=np.einsum('bd,msdij->bmsij',weights,cms,optimize=True)
        wf=cm_metrics(drawn)[0].mean(axis=-1)
        results.append(wf)
    return np.concatenate(results,axis=0)


def report(base,seeds,config,signature):
    base=Path(base);out=base/'reports';out.mkdir(parents=True,exist_ok=True)
    runs=[];rows={};canonical=None
    for seed in seeds:
        marker=base/'scores'/f'seed{seed}_s7r_complete.json'
        if not marker.exists():raise FileNotFoundError(f'Missing complete evaluation: {marker}')
        if json.loads(marker.read_text())['run_signature']!=signature:
            raise ValueError('Evaluation signature mismatch')
        runs.append(json.loads((base/'scores'/f'seed{seed}_validation.json').read_text()))
        for method in METHODS:
            if method=='entropy_router':continue  # legacy offline mixture has no separate raw JSONL
            rs=read_jsonl(base/'scores'/f'seed{seed}_{method}_validation.jsonl')
            identity=[(r['key'],r['dialogue_id'],r['gold']) for r in rs]
            if canonical is None:canonical=identity
            if identity!=canonical:raise ValueError(f'Prediction alignment mismatch: {seed} {method}')
            rows[seed,method]=rs
    if not canonical:raise ValueError('No validation targets')
    y=np.array([LABELS.index(x[2]) for x in canonical]);dialogues=[x[1] for x in canonical]
    cms=[];metrics={};predictions={}
    for method in METHODS:
        cs=[];method_results=[]
        for seed,run in zip(seeds,runs):
            if run['labels']!=LABELS or run['targets']!=len(canonical):raise ValueError('Label/target mismatch')
            m=run['variants'][method]
            pred=np.array([LABELS.index(x) for x in m['predictions']])
            if len(pred)!=len(y):raise ValueError('Seed prediction count mismatch')
            if method!='entropy_router':
                rawpred=np.array([int(np.argmax(r['scores'])) for r in rows[seed,method]])
                if not np.array_equal(pred,rawpred):raise ValueError('Raw/summary prediction mismatch')
            cm,ds=confusion(y,pred,dialogues)
            wf,macro,acc,per=cm_metrics(cm.sum(axis=0))
            if abs(float(wf)-m['weighted_f1'])>1e-9:raise ValueError('wF1 recomputation mismatch')
            method_results.append({'seed':seed,'weighted_f1':float(wf),'macro_f1':float(macro),
                'accuracy':float(acc),'per_emotion_f1':per.tolist(),
                **{k:float(m[k]) for k in ['nll','brier','ece','temperature']}})
            predictions[method,seed]=pred;cs.append(cm)
        cms.append(cs)
        metrics[method]={'name':NAMES[method],'seeds':method_results}
        for name in ['weighted_f1','macro_f1','accuracy','nll','brier','ece','temperature']:
            a=np.array([r[name] for r in method_results]);metrics[method][name]={'mean':float(a.mean()),'sd':float(a.std(ddof=1))}
    cms=np.asarray(cms)
    print('[report] paired dialogue bootstrap; metric = mean per-seed F1',flush=True)
    boot=paired_bootstrap(cms,int(config.get('bootstrap_samples',10000)),int(config.get('bootstrap_seed',20261021)))
    contrasts={}
    pairs=[('full_supervised','generic_reasoning'),('full_supervised','shuffled_supervised'),
           ('full_supervised','evidence_supervised'),('full_supervised','full_no_impact'),
           ('full_supervised','current_only'),('full_supervised','always_recent3'),
           ('mechanism_selected','mechanism_current'),('mechanism_selected','mechanism_recent3')]
    for a,b in pairs:
        i,j=METHODS.index(a),METHODS.index(b)
        delta=metrics[a]['weighted_f1']['mean']-metrics[b]['weighted_f1']['mean']
        diffs=boot[:,i]-boot[:,j]
        contrasts[f'{a}_minus_{b}']={'point_delta':delta,'ci95':np.quantile(diffs,[.025,.975]).tolist(),
            'metric':'mean_seed_weighted_f1','sampling_unit':'dialogue','ensemble_used':False,
            'primary':(a,b)==('full_supervised','generic_reasoning')}
    # Two-sided paired randomization, sharing each dialogue swap across all seeds.
    primary=contrasts['full_supervised_minus_generic_reasoning']
    i,j=METHODS.index('full_supervised'),METHODS.index('generic_reasoning')
    rng=np.random.default_rng(20261022);extreme=0;nperm=int(config.get('permutation_samples',100000))
    delta_cm=cms[i]-cms[j];threshold=abs(primary['point_delta'])-1e-15
    for start in range(0,nperm,250):
        mask=rng.integers(0,2,size=(min(250,nperm-start),len(ds)))
        change=np.einsum('bd,sdij->bsij',mask,delta_cm,optimize=True)
        a=cms[i].sum(axis=1)[None,...]-change;b=cms[j].sum(axis=1)[None,...]+change
        diff=(cm_metrics(a)[0]-cm_metrics(b)[0]).mean(axis=-1)
        extreme+=int(np.sum(np.abs(diff)>=threshold))
        if (start+250)%25000==0:print(f'[report] primary paired permutations {min(start+250,nperm)}/{nperm}',flush=True)
    primary['permutation_p_two_sided']=(extreme+1)/(nperm+1)
    primary['permutation_samples']=nperm
    with (out/'per_emotion_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['method','emotion','support','precision_mean','recall_mean','f1_mean','f1_seed_sd'])
        for mi,method in enumerate(METHODS):
            cm=cms[mi].sum(axis=1);tp=np.diagonal(cm,axis1=-2,axis2=-1)
            pr=np.divide(tp,cm.sum(axis=-2),out=np.zeros_like(tp,dtype=float),where=cm.sum(axis=-2)>0)
            re=np.divide(tp,cm.sum(axis=-1),out=np.zeros_like(tp,dtype=float),where=cm.sum(axis=-1)>0)
            f1=cm_metrics(cm)[3]
            for li,l in enumerate(LABELS):w.writerow([method,l,int(np.sum(y==li)),pr[:,li].mean(),re[:,li].mean(),f1[:,li].mean(),f1[:,li].std(ddof=1)])
    with (out/'corrections_and_harms.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['seed','comparison','corrected','harmed','targets'])
        for seed in seeds:
            for a,b in pairs:
                pa,pb=predictions[a,seed],predictions[b,seed]
                w.writerow([seed,a+'_minus_'+b,int(np.sum((pa==y)&(pb!=y))),int(np.sum((pa!=y)&(pb==y))),len(y)])
    write_json(out/'confusion_matrices.json',{'labels':LABELS,'seeds':seeds,
        'matrices':{m:[c.tolist() for c in cms[i].sum(axis=1)] for i,m in enumerate(METHODS)}})
    result={'experiment':'MELD Stage7R 1500','seeds':seeds,'targets':len(y),'dialogues':len(ds),
            'methods':metrics,'comparisons':contrasts,'test_locked':True,'run_signature':signature,
            'bootstrap':{'samples':len(boot),'seed':int(config.get('bootstrap_seed',20261021)),
                'estimand':'mean of per-seed F1 differences; no probability ensemble; all 7 labels',
                'note':'CI conditional on fixed training seeds; seed SD is reported separately'},
            'primary_advantage_supported_on_dev':primary['ci95'][0]>0 and primary['permutation_p_two_sided']<.05}
    write_json(out/'validation_summary.json',result);write_json(out/'paired_dialogue_bootstrap.json',contrasts)
    lines=['# MELD Stage7R — 1,500条重新标注验证','',
        '> 1,200 teacher SFT targets; 300 held-out teacher targets; 5 fixed seeds; MELD dev only; test locked.','',
        '| 方法 | wF1 | Macro F1 | Accuracy |','|---|---:|---:|---:|']
    for m in METHODS:
        cells=[f"{metrics[m][x]['mean']:.4f} ± {metrics[m][x]['sd']:.4f}" for x in ['weighted_f1','macro_f1','accuracy']]
        lines.append('| '+NAMES[m]+' | '+' | '.join(cells)+' |')
    lines+=['','| 比较 | 平均种子 wF1 差值 | 对话级95% CI |','|---|---:|---|']
    for (a,b),v in zip(pairs,contrasts.values()):lines.append(f"| {NAMES[a]} − {NAMES[b]} | {v['point_delta']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    lines+=['',f"唯一主比较双侧配对置换 p={primary['permutation_p_two_sided']:.6f}。其他比较为探索性，CI未作多重比较校正。",'',
        '完整adapter的三个机制路径使用同一中性分类提示，只改变提供的历史；所选证据来自已经缓存的预测计划。',
        '基础情绪模型和各继续训练分支是不同adapter，跨adapter比较不单独归因于历史选择。',
        '种子SD使用ddof=1。Bootstrap对同一批对话同步重采样，在每个种子上重新计算F1，再平均差值；没有平均概率形成ensemble。',
        'dev已经用于前期开发；本轮仍是开发集验证。教师自审不是独立人工验证，不作推理速度声明。',
        '每类结果：per_emotion_summary.csv；纠错/引错：corrections_and_harms.csv；机制诊断：controller_diagnostics.json。',
        '主比较CI包含零时，不能宣称额外结构监督优于仅情绪标签对照，也不能解释为二者等效。MELD test保持封存。']
    (out/'validation_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out/'validation_summary.md');return result
