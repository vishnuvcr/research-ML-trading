import json
from datetime import datetime, timezone
from pathlib import Path
import argparse

ROOT=Path(__file__).resolve().parents[2]

p=argparse.ArgumentParser()
p.add_argument('--phase',required=True)
p.add_argument('--step',required=True)
p.add_argument('--status',required=True)
p.add_argument('--summary',required=True)
p.add_argument('--error',default='')
a=p.parse_args()

now=datetime.now(timezone.utc).isoformat()
entry={'timestamp':now,'phase':a.phase,'step_id':a.step,'status':a.status,'summary':a.summary}
with (ROOT/'research/RUN_LOG.jsonl').open('a',encoding='utf-8') as f:
    f.write(json.dumps(entry,ensure_ascii=False)+'\n')

status_path=ROOT/'research/STATUS.json'
status=json.loads(status_path.read_text(encoding='utf-8'))
status['phases'][a.phase]['status']=a.status
status['phases'][a.phase]['last_step']=a.step
status['active_phase']=a.phase
status['last_updated']=now
status_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')

if a.error:
    with (ROOT/'research/ERROR_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## {now} — {a.phase}/{a.step}\n\nStatus: {a.status}\n\n{a.error}\n')

print(json.dumps(entry,ensure_ascii=False))
