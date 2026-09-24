"""Character n-gram learned router; graph remains the source of every verdict."""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from claim_graph_gate import decide

ROOT=Path(__file__).resolve().parent
V1=ROOT/'data'/'claim_router_dataset_v0.1.json'
V2=ROOT/'data'/'claim_router_v0.2_plan.json'
PLAN=ROOT/'data'/'claim_router_v0.3_plan.json'

def features(text):
    normalized=re.sub(r'holdou[a-ząćęłńóśźż]*','holdout',text.lower())
    compact=' '+re.sub(r'[^a-ząćęłńóśźż]+',' ',normalized)+' '
    return Counter(compact[i:i+n] for n in (3,4,5) for i in range(len(compact)-n+1))

def train(rows):
    profiles=defaultdict(Counter)
    for text,label in rows: profiles[label].update(features(text))
    norms={label:math.sqrt(sum(value*value for value in profile.values())) for label,profile in profiles.items()}
    return {'profiles':{label:dict(profile) for label,profile in profiles.items()},'norms':norms}

def predict(model,question,threshold=0.12):
    query=features(question); qnorm=math.sqrt(sum(value*value for value in query.values())) or 1.0; scores={}
    for label,raw in model['profiles'].items():
        dot=sum(value*raw.get(token,0) for token,value in query.items()); scores[label]=dot/(qnorm*model['norms'][label]) if model['norms'][label] else 0.0
    label,score=max(scores.items(),key=lambda pair:pair[1]); return (label if score>=threshold else 'INCONCLUSIVE_NO_MATCH'),score

def guarded_route(model,question):
    learned,similarity=predict(model,question); logical=decide(question); logical_label=logical.node_id or 'INCONCLUSIVE_NO_MATCH'
    return {'learned_label':learned,'similarity':similarity,'logical_label':logical_label,'agreement':learned==logical_label,'verdict':logical.verdict}
