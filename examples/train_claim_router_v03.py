import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from claim_router_learning_v03 import PLAN,V1,V2,guarded_route,train
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v1=json.loads(V1.read_text(encoding='utf-8'));v2=json.loads(V2.read_text(encoding='utf-8'));plan=json.loads(PLAN.read_text(encoding='utf-8'))
model=train(v1['train']+v1['calibration']+v2['extra_train']+v2['calibration']);rows=[]
for q,expected in plan['calibration']:
 r={'question':q,'expected':expected,**guarded_route(model,q)};r['learned_correct']=r['learned_label']==expected;rows.append(r)
accuracy=sum(r['learned_correct'] for r in rows)/len(rows);agreement=sum(r['agreement'] for r in rows)/len(rows)
report={'status':'CALIBRATION_PASS_HOLDOUT_UNREAD' if accuracy==1 and agreement==1 else 'CALIBRATION_FAIL_HOLDOUT_BLOCKED','plan_sha256':sha(PLAN),'learned_accuracy':accuracy,'logic_agreement':agreement,'rows':rows,'holdout_accessed':False}
out=ROOT/'learning_runs'/f"claim_router_v03_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}";out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':report['status'],'learned_accuracy':accuracy,'logic_agreement':agreement,'holdout_accessed':False,'report':str(out/'report.json')},ensure_ascii=False,indent=2))
