import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STATUS=ROOT/'research/STATUS.json'
DOCS=ROOT/'docs'

status=json.loads(STATUS.read_text(encoding='utf-8'))
DOCS.mkdir(exist_ok=True)
(DOCS/'research_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')

rows=[]
for phase,data in status['phases'].items():
    rows.append(f"<tr><td>{phase}</td><td>{data['status']}</td><td>{data.get('last_step') or ''}</td></tr>")
html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Research Control Center</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}</style></head><body>'''
html+=f"<h1>Research Control Center</h1><p><b>Program:</b> {status['program']}<br><b>Protocol:</b> {status['protocol_version']}<br><b>Status:</b> {status['overall_status']}</p>"
html+='<table><tr><th>Phase</th><th>Status</th><th>Last step</th></tr>'+''.join(rows)+'</table>'
html+="<p><a href='research_status.json'>Machine-readable status</a></p></body></html>"
(DOCS/'index.html').write_text(html,encoding='utf-8')
print('GitHub Pages artifacts updated')
