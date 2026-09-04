#!/usr/bin/env python3
"""Local PiPER collection console with ROS services, logs and previews."""
import json, os, signal, socket, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "lerobot_piper/script/direct_collect.sh"
SERVICES = ROOT / "lerobot_piper/script/start_robot_services.sh"
PREVIEW = ROOT / "lerobot_piper/script/ros_preview_server.py"
ALOHA = "/home/agilex/miniconda3/envs/aloha/bin/python"
ROS_SETUP = "source /opt/ros/noetic/setup.bash; source /home/agilex/cobot_magic/camera_ws/devel/setup.bash; source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
DEFAULT_DATASET = ROOT / "data/piper_lerobot_direct_v2"
services_proc = preview_proc = capture_proc = None
preview_external = False
services_log = Path("/tmp/piper_robot_services")
capture_log = Path("/tmp/piper_direct_capture.log")
control_file = Path("/tmp/piper_collect_control")
state_file = Path("/tmp/piper_collect_control.state")
episode_file = Path("/tmp/piper_collect_control.episode")
capture_config = None

HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PiPER Collect</title><style>
:root{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#252525;background:#fff}*{box-sizing:border-box}body{margin:0}.wrap{max-width:1320px;margin:auto;padding:34px 28px 70px}.top{display:flex;justify-content:space-between;margin-bottom:28px}.brand{font-size:23px;letter-spacing:-.04em}.muted{color:#777}.layout{display:grid;grid-template-columns:300px 1fr;gap:24px}.panel{border:1px solid #e9e7e3;border-radius:14px;padding:18px;background:#fff;box-shadow:0 4px 18px #00000008}h2{font-size:17px;margin:0 0 14px}label{display:block;font-size:12px;color:#777;margin:11px 0 5px}input,select{width:100%;border:1px solid #dedbd6;border-radius:8px;padding:9px;font-size:13px}button{border:0;border-radius:8px;padding:10px 13px;font-size:13px;cursor:pointer}.primary{background:#252525;color:#fff;width:100%;margin-top:17px}.secondary{background:#f0eee9;width:100%;margin-top:8px}.status{font-size:13px;margin-top:13px;line-height:1.5}.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.cam{border-radius:10px;overflow:hidden;background:#f2f0ed}.cam img{display:block;width:100%;aspect-ratio:16/10;object-fit:cover}.cam div{font-size:12px;padding:7px 9px}.monitor{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:15px}.log{height:170px;overflow:auto;background:#202020;color:#d9f5dc;padding:11px;border-radius:9px;font:11px/1.45 monospace;white-space:pre-wrap}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.videos video{width:100%;background:#eee;border-radius:9px}.video-name{font-size:11px;color:#777;margin-top:4px}.joint{width:100%;height:240px;border:1px solid #ece9e5;border-radius:10px;background:#faf9f7}.hint{font-size:12px;color:#777;margin:8px 0}.kbd{font-family:monospace;background:#f0eee9;border-radius:4px;padding:2px 5px}@media(max-width:900px){.layout{grid-template-columns:1fr}.cams,.videos{grid-template-columns:1fr 1fr}.monitor{grid-template-columns:1fr}}@media(max-width:560px){.cams,.videos{grid-template-columns:1fr}}</style></head><body><main class="wrap"><div class="top"><div class="brand">PiPER / collect</div><div class="muted">direct LeRobot v3 · 960×540</div></div><div class="layout"><aside class="panel"><h2>采集设置</h2><label>数据集目录</label><input id="dataset" value="/home/agilex/wxwu/data/piper_lerobot_direct_v2"><label>Repo ID</label><input id="repo" value="local/piper_dual_arm"><label>任务</label><input id="task" value="dual-arm manipulation"><label>Episode</label><input id="episode" type="number" min="0" value="0"><label>最大帧数</label><input id="frames" type="number" min="1" value="3000"><label>FPS</label><input id="fps" type="number" min="1" value="30"><label>相机</label><select id="res"><option>960x540</option><option>640x480</option></select><button class="primary" onclick="start()">开始采集 <span class="kbd">Space</span></button><button class="secondary" onclick="stop()">停止当前 episode</button><button class="secondary" onclick="retry()">重试 / 重启相机与机械臂</button><div class="status" id="status">正在启动服务…</div></aside><section><div class="panel"><h2>实时相机</h2><div class="cams"><div class="cam"><img src="http://127.0.0.1:8766/stream/front"><div>front</div></div><div class="cam"><img src="http://127.0.0.1:8766/stream/left"><div>left</div></div><div class="cam"><img src="http://127.0.0.1:8766/stream/right"><div>right</div></div></div><div class="monitor"><div><h2>运行日志</h2><pre class="log" id="log">等待日志…</pre></div><div><h2>监测</h2><pre class="log" id="monitor">services: starting</pre></div></div></div><div class="panel" style="margin-top:18px"><h2>最近完成的 episode</h2><div id="resultMeta" class="hint">尚无已完成 episode</div><div id="videos" class="videos"></div><div class="hint">关节轨迹（左右臂各 7 个关节，简化三维连杆显示）</div><canvas id="joint" class="joint" width="900" height="240"></canvas></div></section></div></main><script>
let running=false,logFollow=true;async function api(p,o){let r=await fetch(p,o);return r.json()}function v(id){return document.getElementById(id).value}async function start(){let b={dataset_path:v('dataset'),repo_id:v('repo'),task:v('task'),episode_idx:+v('episode'),timesteps:+v('frames'),fps:+v('fps'),camera_resolution:v('res')};let d=await api('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});if(d.episode_idx!==undefined)document.getElementById('episode').value=d.episode_idx;document.getElementById('status').textContent=d.message||d.error}async function stop(){let d=await api('/api/stop',{method:'POST'});document.getElementById('status').textContent=d.message||d.error}async function retry(){let d=await api('/api/retry',{method:'POST'});document.getElementById('status').textContent=d.message||d.error}async function refresh(){let s=await api('/api/status');running=s.running;if(s.capture==='waiting'&&s.next_episode!==null)document.getElementById('episode').value=s.next_episode;document.getElementById('status').textContent=s.message;let log=document.getElementById('log');log.textContent=s.log;if(logFollow)log.scrollTop=log.scrollHeight;document.getElementById('monitor').textContent=`services: ${s.services}\npreview: ${s.preview}\ncapture: ${s.capture}`;let e=await api('/api/episode?dataset='+encodeURIComponent(v('dataset'))+'&episode='+v('episode'));document.getElementById('resultMeta').textContent=e.meta||'';document.getElementById('videos').innerHTML=(e.videos||[]).map(x=>`<div><video controls preload="metadata" src="${x.url}"></video><div class="video-name">${x.name}</div></div>`).join('');draw(e.states||[])}function draw(a){let c=document.getElementById('joint'),x=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight;x.clearRect(0,0,w,h);if(!a.length)return;let step=Math.max(1,Math.floor(a.length/80));for(let arm=0;arm<2;arm++){x.strokeStyle=arm?'#9a6b45':'#4e6f9e';x.lineWidth=2;for(let i=0;i<a.length;i+=step){let q=a[i].slice(arm*7,arm*7+7),px=arm?w*.72:w*.28,py=h*.5;x.beginPath();x.moveTo(px,py);for(let j=0;j<7;j++){let len=18-j*1.2,ang=q[j]+j*.35,nx=px+len*Math.cos(ang),ny=py+len*Math.sin(ang);x.lineTo(nx,ny);px=nx;py=ny}x.stroke()}}}document.getElementById('log').addEventListener('scroll',e=>{let el=e.target;logFollow=el.scrollTop+el.clientHeight>=el.scrollHeight-8});document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target.tagName!=='INPUT'){e.preventDefault();running?stop():start()}});setInterval(refresh,1200);refresh();</script></body></html>'''

def tail_logs():
    out=[]
    for p in [services_log/'services.log',services_log/'roscore.log',services_log/'cameras.log',services_log/'arms.log',services_log/'preview.log',capture_log]:
        if p.exists(): out += [f"\n### {p.name}"] + p.read_text(errors='replace').splitlines()[-100:]
    return '\n'.join(out)[-24000:]


def capture_state():
    if state_file.exists():
        return state_file.read_text(encoding='utf-8').strip()
    return 'idle'


def send_control(command):
    with control_file.open('w', encoding='utf-8') as f:
        f.write(command + '\n')

def start_services():
    global services_proc, preview_proc, preview_external
    services_log.mkdir(parents=True, exist_ok=True)
    if services_proc is None or services_proc.poll() is not None:
        env=os.environ.copy(); env['PIPER_CAMERA_RESOLUTION']='960x540'; env['PIPER_SERVICE_LOG_DIR']=str(services_log)
        services_proc=subprocess.Popen([str(SERVICES)],env=env,start_new_session=True,stdout=(services_log/'services.log').open('a'),stderr=subprocess.STDOUT)
    if preview_proc is None or preview_proc.poll() is not None:
        try:
            with socket.create_connection(('127.0.0.1', 8766), timeout=0.3):
                preview_external = True
        except OSError:
            preview_external = False
            preview_cmd=f"{ROS_SETUP}; exec '{ALOHA}' '{PREVIEW}'"
            preview_proc=subprocess.Popen(['bash','-lc',preview_cmd],start_new_session=True,stdout=(services_log/'preview.log').open('w'),stderr=subprocess.STDOUT)

class Handler(BaseHTTPRequestHandler):
    def _json(self,d,code=200):
        raw=json.dumps(d,ensure_ascii=False).encode(); self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        global capture_proc
        p=urlparse(self.path)
        if p.path=='/':
            raw=HTML.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw); return
        if p.path=='/api/status':
            preview_up = preview_external or (preview_proc is not None and preview_proc.poll() is None)
            state = capture_state()
            active = capture_proc is not None and capture_proc.poll() is None
            capture_status = 'running' if active and state == 'recording' else ('waiting' if active and state == 'waiting' else ('starting' if active else 'idle'))
            message = '采集中…' if capture_status == 'running' else ('等待下一条 episode（按空格开始）' if capture_status == 'waiting' else ('正在保存当前 episode…' if state == 'saving' else ('正在初始化同步采集…' if capture_status == 'starting' else '服务就绪')))
            next_episode = None
            if episode_file.exists():
                try: next_episode = int(episode_file.read_text(encoding='utf-8'))
                except ValueError: pass
            self._json({'running':capture_status == 'running','services':'up' if services_proc is not None and services_proc.poll() is None else 'down','preview':'up' if preview_up else 'down','capture':capture_status,'message':message,'next_episode':next_episode,'log':tail_logs()}); return
        if p.path in ('/api/episode','/api/previews'):
            # Do not let the preview reader touch Parquet files while the
            # writer is updating them; pyarrow can crash on a partial file.
            if capture_proc is not None and capture_proc.poll() is None and capture_state() == 'recording':
                self._json({'meta':'当前 episode 采集中，完成后自动刷新','videos':[],'states':[],'items':[]}); return
            q=parse_qs(p.query); root=Path(q.get('dataset',[str(DEFAULT_DATASET)])[0]).expanduser().resolve(); ep=int(q.get('episode',[0])[0]); items=[]; states=[]
            wanted=f'file-{ep:03d}.mp4'
            for f in sorted(root.glob('videos/**/*.mp4')):
                if f.name != wanted and any(root.glob('videos/**/file-*.mp4')): continue
                items.append({'name':str(f.relative_to(root)),'url':'/media/'+quote(str(f.relative_to(root)))+'?dataset='+quote(str(root))})
            self._json({'meta':f'episode {ep} · {len(states)} 个关节采样 · {len(items)} 个视频','videos':items,'states':states,'items':items}); return
        if p.path.startswith('/media/'):
            q=parse_qs(p.query); root=Path(q.get('dataset',[str(DEFAULT_DATASET)])[0]).expanduser().resolve(); path=(root/unquote(p.path[7:])).resolve()
            if root not in path.parents or not path.is_file(): self.send_error(404); return
            data=path.read_bytes(); self.send_response(200); self.send_header('Content-Type','video/mp4'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)
    def do_POST(self):
        global capture_proc, capture_config
        if self.path=='/api/start':
            if capture_proc is not None and capture_proc.poll() is None:
                if capture_state() == 'waiting':
                    send_control('start'); self._json({'message':'下一条 episode 已开始'}); return
                self._json({'error':'已有采集在运行'},409); return
            b=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
            info=Path(b['dataset_path']).expanduser().resolve()/'meta'/'info.json'
            if info.exists():
                b['episode_idx']=int(json.loads(info.read_text(encoding='utf-8')).get('total_episodes',0))
            else:
                b['episode_idx']=0
            control_file.unlink(missing_ok=True); state_file.unlink(missing_ok=True)
            out=capture_log.open('w'); args=[str(DIRECT),'--skip-can','--yes','--auto-start','--continuous']
            for k in ('dataset_path','repo_id','task','episode_idx','timesteps','fps','camera_resolution'): args += ['--'+k.replace('_','-'),str(b[k])]
            env=os.environ.copy(); env['PIPER_CONTROL_FILE']=str(control_file)
            capture_config=b
            capture_proc=subprocess.Popen(args,cwd=ROOT,env=env,start_new_session=True,stdout=out,stderr=subprocess.STDOUT); self._json({'message':f"episode {b['episode_idx']} 已启动，正在等待同步帧",'episode_idx':b['episode_idx']}); return
        if self.path=='/api/stop':
            if capture_proc is not None and capture_proc.poll() is None: send_control('stop'); self._json({'message':'已请求停止当前 episode'})
            else: self._json({'message':'当前没有运行中的采集'}); return
        if self.path=='/api/retry':
            for p in (services_proc,preview_proc):
                if p is not None and p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
            time.sleep(1); start_services(); self._json({'message':'已重启 roscore、相机、双臂和预览服务'}); return
        self.send_error(404)

start_services()
if __name__=='__main__':
    port=int(os.environ.get('PIPER_COLLECT_WEB_PORT','8765')); print(f'http://127.0.0.1:{port}',flush=True); ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
