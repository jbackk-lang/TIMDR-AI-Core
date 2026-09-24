import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from claim_router_learning_v03 import V1,V2,train,guarded_route
PLAN=ROOT/'data'/'claim_router_v0.4_plan.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v1=json.loads(V1.read_text(encoding='utf8'));v2=json.loads(V2.read_text(encoding='utf8'));p=json.loads(PLAN.read_text(encoding='utf8'));m=train(v1['train']+v1['calibration']+v2['extra_train']+v2['calibration']);rows=[]
for q,e in p['calibration']:
 r={'question':q,'expected':e,**guarded_route(m,q)};r['learned_correct']=r['learned_label']==e;rows.append(r)
a=sum(x['learned_correct'] for x in rows)/len(rows);g=sum(x['agreement'] for x in rows)/len(rows);report={'status':'CALIBRATION_PASS_HOLDOUT_UNREAD' if a==1 and g==1 else 'CALIBRATION_FAIL_HOLDOUT_BLOCKED','plan_sha256':sha(PLAN),'learned_accuracy':a,'logic_agreement':g,'rows':rows,'holdout_accessed':False};out=ROOT/'learning_runs'/f"claim_router_v04_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}";out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':report['status'],'learned_accuracy':a,'logic_agreement':g,'holdout_accessed':False,'report':str(out/'report.json')},ensure_ascii=False,indent=2))
