import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from claim_router_learning_v02 import PLAN, V1, guarded_route, train

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

v1=json.loads(V1.read_text(encoding="utf-8")); plan=json.loads(PLAN.read_text(encoding="utf-8"))
model=train(v1["train"]+v1["calibration"]+plan["extra_train"])
rows=[]
for question,expected in plan["calibration"]:
    row={"question":question,"expected":expected,**guarded_route(model,question)}; row["learned_correct"]=row["learned_label"]==expected; rows.append(row)
accuracy=sum(row["learned_correct"] for row in rows)/len(rows); agreement=sum(row["agreement"] for row in rows)/len(rows)
report={"status":"CALIBRATION_PASS_HOLDOUT_UNREAD" if accuracy==1.0 and agreement==1.0 else "CALIBRATION_FAIL_HOLDOUT_BLOCKED","plan_sha256":sha(PLAN),"v1_dataset_sha256":sha(V1),"learned_accuracy":accuracy,"logic_agreement":agreement,"rows":rows,"holdout_accessed":False}
out=ROOT/"learning_runs"/f"claim_router_v02_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; out.mkdir(parents=True,exist_ok=True); (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"status":report["status"],"learned_accuracy":accuracy,"logic_agreement":agreement,"holdout_accessed":False,"report":str(out/"report.json")},ensure_ascii=False,indent=2))
