#!/usr/bin/env python3
"""Local PiPER collection console with ROS services, logs and previews."""
import atexit, concurrent.futures, fcntl, json, os, signal, socket, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "lerobot_piper/script/direct_collect.sh"
SERVICES = ROOT / "lerobot_piper/script/start_robot_services.sh"
PREVIEW = ROOT / "lerobot_piper/script/ros_preview_server.py"
STREAM = ROOT / "lerobot_piper/script/ros_lerobot_stream.py"
RUNNER = ROOT / "lerobot_piper/script/run_collect_web.sh"
LAUNCH_FILE = Path("/home/agilex/cobot_magic/tmp/three_cameras_60hz.launch")
ALOHA = "/home/agilex/miniconda3/envs/aloha/bin/python"
ROS_SETUP = "source /opt/ros/noetic/setup.bash; source /home/agilex/cobot_magic/camera_ws/devel/setup.bash; source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
DEFAULT_DATASET = ROOT / "data/piper_lerobot_direct_v2"
DATASET_ROOT = ROOT / "data"
services_proc = preview_proc = capture_proc = None
preview_external = False
services_log = Path("/tmp/piper_robot_services")
capture_log = Path("/tmp/piper_direct_capture.log")
control_file = Path(f"/tmp/piper_collect_control.{os.getpid()}")
state_file = Path(f"{control_file}.state")
episode_file = Path(f"{control_file}.episode")
capture_config = None
capture_guard = threading.Lock()
cleanup_guard = threading.Lock()
cleanup_started = False
exit_requested = False
web_lock = None

# Every process in this list belongs to this collection UI.  Cleanup is done
# by process group so that wrappers and their children cannot survive a web
# process restart.  The roslaunch/roscore paths are deliberately specific to
# the robot services used here; unrelated system processes are not matched.
RUNTIME_PROCESS_PATTERNS = (
    str(SERVICES),
    str(PREVIEW),
    str(DIRECT),
    str(STREAM),
    "/opt/ros/noetic/bin/roslaunch piper start_ms_piper.launch",
    f"/opt/ros/noetic/bin/roslaunch {LAUNCH_FILE}",
    "/opt/ros/noetic/bin/roslaunch realsense2_camera multi_camera.launch",
    "/opt/ros/noetic/bin/roscore",
)
REQUIRED_TOPICS = (
    "/camera_f/color/image_raw",
    "/camera_l/color/image_raw",
    "/camera_r/color/image_raw",
    "/master/joint_left",
    "/master/joint_right",
    "/puppet/joint_left",
    "/puppet/joint_right",
)
topic_readiness_error = ''

HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PiPER Collect</title><style>
:root{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#252525;background:#fff}*{box-sizing:border-box}body{margin:0}.wrap{max-width:1320px;margin:auto;padding:34px 28px 70px}.top{display:flex;justify-content:space-between;margin-bottom:28px}.brand{font-size:23px;letter-spacing:-.04em}.muted{color:#777}.layout{display:grid;grid-template-columns:minmax(0,300px) minmax(0,1fr);gap:24px}.layout>section{min-width:0}.panel{border:1px solid #e9e7e3;border-radius:14px;padding:18px;background:#fff;box-shadow:0 4px 18px #00000008}h2{font-size:17px;margin:0 0 14px}label{display:block;font-size:12px;color:#777;margin:11px 0 5px}input,select{width:100%;border:1px solid #dedbd6;border-radius:8px;padding:9px;font-size:13px}button{border:0;border-radius:8px;padding:10px 13px;font-size:13px;cursor:pointer}.primary{background:#252525;color:#fff;width:100%;margin-top:17px}.secondary{background:#f0eee9;width:100%;margin-top:8px}.danger{background:#f5d9d5;color:#8a2f25;width:100%;margin-top:8px}.status{font-size:13px;margin-top:13px;line-height:1.5}.cams{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;min-width:0}.cam{min-width:0;border-radius:10px;overflow:hidden;background:#f2f0ed}.cam img{display:block;width:100%;max-width:100%;height:auto;aspect-ratio:16/9;object-fit:cover}.cam div{font-size:12px;padding:7px 9px}.monitor{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;margin-top:15px}.log{height:170px;overflow:auto;background:#202020;color:#d9f5dc;padding:11px;border-radius:9px;font:11px/1.45 monospace;white-space:pre-wrap}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;min-width:0}.videos video{display:block;width:100%;max-width:100%;background:#eee;border-radius:9px}.video-name{font-size:11px;color:#777;margin-top:4px}.joint{width:100%;height:240px;border:1px solid #ece9e5;border-radius:10px;background:#faf9f7}.hint{font-size:12px;color:#777;margin:8px 0}.kbd{font-family:monospace;background:#f0eee9;border-radius:4px;padding:2px 5px}@media(max-width:900px){.layout{grid-template-columns:1fr}.cams,.videos{grid-template-columns:repeat(2,minmax(0,1fr))}.monitor{grid-template-columns:1fr}}@media(max-width:560px){.cams,.videos{grid-template-columns:minmax(0,1fr)}}</style></head><body><main class="wrap"><div class="top"><div class="brand">PiPER / collect</div><div class="muted">direct LeRobot v3 · 960×540</div></div><div class="layout"><aside class="panel"><h2>采集设置</h2><label>数据集目录</label><input id="dataset" value="/home/agilex/wxwu/data/piper_lerobot_direct_v2"><label>Repo ID</label><input id="repo" value="local/piper_dual_arm"><label>任务</label><input id="task" value="dual-arm manipulation"><label>Episode</label><input id="episode" type="number" min="0" value="0"><label>最大帧数</label><input id="frames" type="number" min="1" value="3000"><label>FPS</label><input id="fps" type="number" min="1" value="30"><label>相机</label><select id="res"><option>960x540</option><option>640x480</option></select><button class="primary" onclick="start()">开始采集 <span class="kbd">Space</span></button><button class="secondary" onclick="stop()">停止当前 episode</button><button class="danger" onclick="shutdown()">结束整个采集</button><button class="secondary" onclick="retry()">重试 / 重启相机与机械臂</button><div class="status" id="status">正在启动服务…</div></aside><section><div class="panel"><h2>实时相机</h2><div class="cams"><div class="cam"><img src="http://127.0.0.1:8766/stream/front"><div>front</div></div><div class="cam"><img src="http://127.0.0.1:8766/stream/left"><div>left</div></div><div class="cam"><img src="http://127.0.0.1:8766/stream/right"><div>right</div></div></div><div class="monitor"><div><h2>运行日志</h2><pre class="log" id="log">等待日志…</pre></div><div><h2>监测</h2><pre class="log" id="monitor">services: starting</pre></div></div></div><div class="panel" style="margin-top:18px"><h2>最近完成的 episode</h2><div id="resultMeta" class="hint">尚无已完成 episode</div><div id="videos" class="videos"></div><div class="hint">关节轨迹（左右臂各 7 个关节，简化三维连杆显示）</div><canvas id="joint" class="joint" width="900" height="240"></canvas></div></section></div></main><script>
async function loadDatasets(){let d=await api('/api/datasets');let s=document.getElementById('datasetSelect');if(!s){s=document.createElement('select');s.id='datasetSelect';s.style.marginBottom='8px';s.onchange=chooseDataset;let label=document.createElement('label');label.textContent='选择本地 LeRobot v3 数据集';let input=document.getElementById('dataset');input.parentNode.insertBefore(label,input);input.parentNode.insertBefore(s,input)}let current=v('dataset');s.innerHTML='<option value="">手动输入目录</option>'+(d.datasets||[]).map(x=>`<option value="${x.path}" data-fps="${x.fps}" data-episodes="${x.episodes}">${x.name} · ${x.episodes} episodes · ${x.frames} 帧</option>`).join('');let opt=[...s.options].find(x=>x.value===current);if(opt)s.value=current}function chooseDataset(){let s=document.getElementById('datasetSelect'),o=s.options[s.selectedIndex];if(!o||!o.value)return;document.getElementById('dataset').value=o.value;if(o.dataset.fps)document.getElementById('fps').value=o.dataset.fps;document.getElementById('status').textContent=`已选择 ${o.text}，开始后将从下一条 episode 继续追加`}let running=false,logFollow=true;async function api(p,o){let r=await fetch(p,o);return r.json()}function v(id){return document.getElementById(id).value}async function start(){let b={dataset_path:v('dataset'),repo_id:v('repo'),task:v('task'),episode_idx:+v('episode'),timesteps:+v('frames'),fps:+v('fps'),camera_resolution:v('res')};let d=await api('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});if(d.episode_idx!==undefined)document.getElementById('episode').value=d.episode_idx;document.getElementById('status').textContent=d.message||d.error}async function stop(){let d=await api('/api/stop',{method:'POST'});document.getElementById('status').textContent=d.message||d.error}async function shutdown(){if(!confirm('结束整个采集会话？当前 episode 会先保存，已完成数据不会删除。'))return;let d=await api('/api/shutdown',{method:'POST'});document.getElementById('status').textContent=d.message||d.error}async function retry(){let d=await api('/api/retry',{method:'POST'});document.getElementById('status').textContent=d.message||d.error}async function refresh(){let s=await api('/api/status');running=s.running;if(s.capture==='waiting'&&s.next_episode!==null)document.getElementById('episode').value=s.next_episode;document.getElementById('status').textContent=s.message;let log=document.getElementById('log');log.textContent=s.log;if(logFollow)log.scrollTop=log.scrollHeight;document.getElementById('monitor').textContent=`services: ${s.services}\npreview: ${s.preview}\ncapture: ${s.capture}`;let e=await api('/api/episode?dataset='+encodeURIComponent(v('dataset'))+'&episode='+v('episode'));document.getElementById('resultMeta').textContent=e.meta||'';document.getElementById('videos').innerHTML=(e.videos||[]).map(x=>`<div><video controls preload="metadata" src="${x.url}"></video><div class="video-name">${x.name}</div></div>`).join('');draw(e.states||[])}function draw(a){let c=document.getElementById('joint'),x=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight;x.clearRect(0,0,w,h);if(!a.length)return;let step=Math.max(1,Math.floor(a.length/80));for(let arm=0;arm<2;arm++){x.strokeStyle=arm?'#9a6b45':'#4e6f9e';x.lineWidth=2;for(let i=0;i<a.length;i+=step){let q=a[i].slice(arm*7,arm*7+7),px=arm?w*.72:w*.28,py=h*.5;x.beginPath();x.moveTo(px,py);for(let j=0;j<7;j++){let len=18-j*1.2,ang=q[j]+j*.35,nx=px+len*Math.cos(ang),ny=py+len*Math.sin(ang);x.lineTo(nx,ny);px=nx;py=ny}x.stroke()}}}document.getElementById('log').addEventListener('scroll',e=>{let el=e.target;logFollow=el.scrollTop+el.clientHeight>=el.scrollHeight-8});document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target.tagName!=='INPUT'){e.preventDefault();running?stop():start()}});setInterval(refresh,1200);loadDatasets();refresh();</script></body></html>'''

# Runtime page enhancements (kept outside the compact HTML literal above).
HTML = HTML.replace('value="/home/agilex/wxwu/data/piper_lerobot_direct_v2"', 'value="/home/agilex/wxwu/data/ATTACH_CAP_TO_PEN_1"')
HTML = HTML.replace('direct LeRobot v3', 'direct LeRobot v2.1')
HTML = HTML.replace('选择本地 LeRobot v3 数据集', '选择本地 LeRobot v2.1 数据集')
HTML = HTML.replace('<input id="episode" type="number" min="0" value="0">', '<input id="episode" type="number" min="0" value="0" readonly>')
HTML = HTML.replace("if(o.dataset.fps)document.getElementById('fps').value=o.dataset.fps;", "if(o.dataset.fps)document.getElementById('fps').value=o.dataset.fps;if(o.dataset.episodes)document.getElementById('episode').value=o.dataset.episodes;")
HTML = HTML.replace("}function chooseDataset(){let s=document.getElementById('datasetSelect'),o=s.options[s.selectedIndex];if(!o||!o.value)return;document.getElementById('dataset').value=o.value;if(o.dataset.fps)document.getElementById('fps').value=o.dataset.fps;if(o.dataset.episodes)document.getElementById('episode').value=o.dataset.episodes;document.getElementById('status').textContent=", "}function chooseDataset(){let s=document.getElementById('datasetSelect'),o=s.options[s.selectedIndex],input=document.getElementById('dataset');if(!o||!o.value){input.focus();episodeRenderKey='';document.getElementById('status').textContent='请在下方输入数据集目录，然后开始采集';return}input.value=o.value;if(o.dataset.fps)document.getElementById('fps').value=o.dataset.fps;if(o.dataset.episodes)document.getElementById('episode').value=o.dataset.episodes;episodeRenderKey='';document.getElementById('status').textContent=")
HTML = HTML.replace("setInterval(refresh,1200);loadDatasets();refresh();", "document.getElementById('dataset').addEventListener('change',()=>{let s=document.getElementById('datasetSelect');if(s)s.value='';episodeRenderKey='';refreshEpisodes()});setInterval(refresh,1200);setInterval(loadDatasets,5000);loadDatasets();refresh();")
HTML = HTML.replace('</style>', '.episode-row{border-top:1px solid #e9e7e3;padding:14px 0}.episode-head{display:flex;gap:12px;align-items:baseline;margin-bottom:8px}.episode-info{font-size:12px;color:#777;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.episode-videos{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.episode-videos video{width:100%;background:#eee;border-radius:8px}.view-controls{display:flex;gap:8px;align-items:center;margin:10px 0}.view-controls select{width:auto;min-width:120px}</style>', 1)
HTML = HTML.replace('<div class="panel" style="margin-top:18px"><h2>最近完成的 episode</h2><div id="resultMeta" class="hint">尚无已完成 episode</div><div id="videos" class="videos"></div><div class="hint">关节轨迹（左右臂各 7 个关节，简化三维连杆显示）</div><canvas id="joint" class="joint" width="900" height="240"></canvas></div>', '<div class="panel" style="margin-top:18px"><h2>已采集 episode</h2><div class="view-controls">跳转到 episode：<select id="previewCenter"></select></div><div id="episodes"></div><div id="resultMeta" style="display:none"></div><div id="videos" style="display:none"></div><canvas id="joint" class="joint" width="900" height="240" style="display:none"></canvas></div>')
HTML = HTML.replace('<div class="top"><div class="brand">PiPER / collect</div><div class="muted">direct LeRobot v2.1 · 960×540</div></div>', '<div class="top"><div class="brand">PiPER / collect</div><div class="top-actions"><button class="secondary refresh-episodes" onclick="manualEpisodeRefresh()">刷新 episode</button><div class="muted">direct LeRobot v2.1 · 960×540</div></div></div>')
HTML = HTML.replace('</style>', '.top-actions{display:flex;align-items:center;gap:12px}.refresh-episodes{width:auto;margin:0;padding:7px 12px}</style>', 1)
HTML = HTML.replace('</script></body></html>', '''async function refreshEpisodes(){let s=document.getElementById('previewCenter');if(!s)return;let center=Number(s.value||0);let d=await api('/api/episodes?dataset='+encodeURIComponent(v('dataset'))+'&center='+center);if(!s.options.length){for(let i=0;i<d.total;i++){let o=document.createElement('option');o.value=i;o.textContent='episode '+i;s.appendChild(o)}if(d.total)s.value=d.total>2?1:0}if(!episodePreviewPinned&&d.total&&Number(s.value)!==d.total-1){s.value=d.total-1;return refreshEpisodes()}document.getElementById('episodes').innerHTML=(d.episodes||[]).map(e=>`<div class="episode-row"><div class="episode-head"><b>episode ${e.episode}</b><span>${e.length} 帧 · ${e.duration.toFixed(2)} s</span><span class="episode-info" title="${e.task}">${e.task}</span></div><div class="episode-videos">${e.videos.map(x=>`<video controls preload="metadata" src="${x.url}"></video>`).join('')}</div></div>`).join('');document.querySelectorAll('.episode-videos').forEach(row=>{let vs=[...row.querySelectorAll('video')];vs.forEach(video=>{video.onplay=()=>vs.forEach(x=>{if(x!==video){x.currentTime=video.currentTime;x.play()}});video.onpause=()=>vs.forEach(x=>{if(x!==video)x.pause()});video.onseeked=()=>vs.forEach(x=>{if(x!==video&&Math.abs(x.currentTime-video.currentTime)>.08)x.currentTime=video.currentTime})})})}document.getElementById('previewCenter').addEventListener('change',()=>{episodePreviewPinned=true;refreshEpisodes()});setInterval(refreshEpisodes,3000);refreshEpisodes();</script></body></html>''')
HTML = HTML.replace("})})}document.getElementById('previewCenter').addEventListener", "})})}function manualEpisodeRefresh(){episodeRenderKey='';return refreshEpisodes()}document.getElementById('previewCenter').addEventListener")
HTML = HTML.replace("if(d.total)s.value=d.total>2?1:0", "if(d.total){s.value=d.total>2?d.total-1:0;return refreshEpisodes()}")
HTML = HTML.replace("document.getElementById('episodes').innerHTML=", "if(!running)document.getElementById('episode').value=d.total;document.getElementById('episodes').innerHTML=")
HTML = HTML.replace("let running=false,logFollow=true;", "let running=false,logFollow=true,episodeRenderKey='',episodePreviewPinned=false;")
# The new three-episode view is refreshed by refreshEpisodes().  Remove the
# legacy hidden preview request: with v2.1 filenames it selected every MP4 in
# the dataset every 1.2 seconds, spawning hundreds of HTTP threads and making
# the web process appear to exit unexpectedly under load.
HTML = HTML.replace("let e=await api('/api/episode?dataset='+encodeURIComponent(v('dataset'))+'&episode='+v('episode'));document.getElementById('resultMeta').textContent=e.meta||'';document.getElementById('videos').innerHTML=(e.videos||[]).map(x=>`<div><video controls preload=\"metadata\" src=\"${x.url}\"></video><div class=\"video-name\">${x.name}</div></div>`).join('');draw(e.states||[])", "")
HTML = HTML.replace("if(!running)document.getElementById('episode').value=d.total;document.getElementById('episodes').innerHTML=", "let renderKey=v('dataset')+'|'+center+'|'+(d.episodes||[]).map(e=>e.episode+':'+e.length).join(',');if(renderKey===episodeRenderKey)return;episodeRenderKey=renderKey;if(!running)document.getElementById('episode').value=d.total;document.getElementById('episodes').innerHTML=")

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

def _matching_runtime_pids(patterns):
    """Return PIDs in this checkout, excluding this process group."""
    if isinstance(patterns, str):
        patterns = (patterns,)
    own_pgid = os.getpgrp()
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if pid == os.getpid() or os.getpgid(pid) == own_pgid:
                continue
            command = (entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(pattern in command for pattern in patterns):
            result.append(pid)
    return sorted(result)


def active_collector_pids():
    """Find collectors from this checkout that are not owned by this process."""
    return _matching_runtime_pids((str(DIRECT), str(STREAM)))

def stop_process_group(proc, timeout=8):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass

def stop_matching_processes(patterns, timeout=8):
    """Stop all matching process groups, escalating to SIGKILL if needed."""
    pids = _matching_runtime_pids(patterns)
    pgids = set()
    for pid in pids:
        try:
            pgids.add(os.getpgid(pid))
        except (ProcessLookupError, PermissionError):
            pass
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    # Never leave a matched process group behind after a restart.  Refresh
    # the PID list after SIGKILL because a wrapper can fork between scans.
    for _ in range(20):
        if not _matching_runtime_pids(patterns):
            return
        time.sleep(0.1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _matching_runtime_pids(patterns):
            return
        time.sleep(0.1)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def remove_runtime_files():
    """Remove control files and Unix sockets left by an interrupted run."""
    for pattern in ('piper_collect_control.*', 'piper_lerobot_stream.*'):
        for path in Path('/tmp').glob(pattern):
            try:
                if path.is_file() or path.is_socket():
                    path.unlink()
            except (FileNotFoundError, OSError):
                pass


def cleanup_stale_runtime():
    """Fully clear a previous UI/service run before starting a new one."""
    stop_matching_processes(RUNTIME_PROCESS_PATTERNS)
    remove_runtime_files()


def _topic_has_message(topic, probe_timeout=4):
    """Return whether one message can actually be received from *topic*.

    ``rostopic list`` only proves that a publisher registered its name.  A
    camera can still be present in the graph while its USB stream is dead, so
    probe the stream itself before allowing a recording to start.
    """
    try:
        result = subprocess.run(
            ['bash', '-lc',
             f'{ROS_SETUP}; exec timeout {probe_timeout}s rostopic echo -n 1 {topic} >/dev/null 2>&1'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=probe_timeout + 2,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ros_master_available(timeout=0.2):
    """Check the local ROS master without spawning a shell command."""
    try:
        with socket.create_connection(('127.0.0.1', 11311), timeout=timeout):
            return True
    except OSError:
        return False


def topics_ready(timeout=30):
    """Wait until all required topics are advertised and emitting messages."""
    global topic_readiness_error
    topic_readiness_error = ''
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            listed = subprocess.run(
                ['bash', '-lc', f'{ROS_SETUP}; rostopic list'],
                capture_output=True, text=True, timeout=3,
            ).stdout.splitlines()
            missing = [topic for topic in REQUIRED_TOPICS if topic not in listed]
            if missing:
                topic_readiness_error = '未发布: ' + ', '.join(missing)
            else:
                # Probe in parallel so a dead camera does not consume the
                # entire readiness timeout one topic at a time.
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(REQUIRED_TOPICS)) as pool:
                    probes = {topic: pool.submit(_topic_has_message, topic) for topic in REQUIRED_TOPICS}
                    active = [topic for topic, probe in probes.items() if probe.result()]
                    if len(active) == len(REQUIRED_TOPICS):
                        return True
                    topic_readiness_error = '无实际消息: ' + ', '.join(
                        topic for topic in REQUIRED_TOPICS if topic not in active
                    )
        except (OSError, subprocess.SubprocessError):
            pass
        time.sleep(0.5)
    return False

def cleanup_runtime():
    """Save an active episode, then stop every process group owned by the web UI."""
    global capture_proc, preview_proc, services_proc, cleanup_started, web_lock
    with cleanup_guard:
        if cleanup_started:
            return
        cleanup_started = True

    capture = capture_proc
    if capture is not None and capture.poll() is None:
        try:
            send_control('shutdown')
            capture.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            pass
    stop_process_group(capture)
    stop_process_group(preview_proc)
    stop_process_group(services_proc, timeout=12)
    capture_proc = preview_proc = services_proc = None
    remove_runtime_files()
    for path in (control_file, state_file, episode_file):
        path.unlink(missing_ok=True)
    if web_lock is not None:
        try:
            fcntl.flock(web_lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        web_lock.close()
        web_lock = None

def request_exit(_signum, _frame):
    global exit_requested
    if exit_requested:
        return
    exit_requested = True
    raise KeyboardInterrupt

def acquire_web_lock():
    global web_lock
    lock = Path('/tmp/piper_collect_web.lock').open('a+', encoding='utf-8')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit('collect_web 已经在运行，拒绝启动第二个实例')
    lock.seek(0); lock.truncate(); lock.write(f'{os.getpid()}\n'); lock.flush()
    web_lock = lock

def list_datasets():
    """Return every local dataset directory, including newly-created empty ones.

    The previous implementation only returned directories with a complete
    LeRobot ``meta/info.json`` (and a specific codebase version).  That made a
    freshly-created directory such as ``data/MAKE_HAM`` invisible until after
    its first episode had been written.  The UI needs to be able to select the
    directory before collection starts, so metadata is now optional.
    """
    if not DATASET_ROOT.exists():
        return []
    root_base = DATASET_ROOT.resolve()
    roots = {p.resolve() for p in DATASET_ROOT.iterdir() if p.is_dir()}
    # Also retain nested datasets discovered from metadata, for compatibility
    # with existing layouts that place datasets below a grouping directory.
    roots.update(info_path.parent.parent.resolve() for info_path in DATASET_ROOT.rglob("meta/info.json"))

    result = []
    for root in sorted(roots, key=lambda p: str(p)):
        info = {}
        info_path = root / "meta" / "info.json"
        try:
            if info_path.is_file():
                parsed = json.loads(info_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    info = parsed
        except (OSError, ValueError, json.JSONDecodeError):
            info = {}
        try:
            name = str(root.relative_to(root_base))
        except ValueError:
            name = root.name
        result.append({
            "name": name,
            "path": str(root),
            "episodes": int(info.get("total_episodes", 0) or 0),
            "frames": int(info.get("total_frames", 0) or 0),
            "fps": int(info.get("fps", 30) or 30),
        })
    return result

def read_episode_rows(root):
    """Build the completed episode view without loading PyArrow.

    PyArrow 25's native ``libarrow.so`` crashes when several HTTP threads
    inspect Parquet metadata concurrently on this machine.  episodes.jsonl is
    written only after an episode is committed, so it is both sufficient and
    safer as the source of truth for this UI.
    """
    info_path = root / 'meta/info.json'
    episodes_path = root / 'meta/episodes.jsonl'
    try:
        info = json.loads(info_path.read_text(encoding='utf-8'))
        fps = float(info.get('fps', 30))
        records = {}
        for line in episodes_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record['episode_index'])] = record
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return []
    rows = []
    for episode, record in sorted(records.items()):
        length = int(record.get('length', 0))
        if length <= 0:
            continue
        chunk = episode // 1000
        parquet = root / 'data' / f'chunk-{chunk:03d}' / f'episode_{episode:06d}.parquet'
        if not parquet.is_file():
            continue
        videos = []
        for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            rel = Path('videos') / f'chunk-{chunk:03d}' / f'observation.images.{key}' / f'episode_{episode:06d}.mp4'
            if (root / rel).is_file():
                videos.append({'name': key, 'url': '/media/' + quote(str(rel)) + '?dataset=' + quote(str(root))})
        rows.append({'episode': episode, 'length': length, 'duration': length / fps,
                     'task': (record.get('tasks') or [''])[0], 'videos': videos})
    return rows

def start_services():
    global services_proc, preview_proc, preview_external
    # Every start/retry has an explicit clean boundary.  Stop process groups
    # before spawning the replacement wrapper; otherwise an old roslaunch or
    # preview can survive long enough to race the new ROS graph.
    if services_proc is not None and services_proc.poll() is None:
        stop_process_group(services_proc, timeout=12)
    if preview_proc is not None and preview_proc.poll() is None:
        stop_process_group(preview_proc)
    services_proc = preview_proc = None
    preview_external = False
    # Do not call cleanup_stale_runtime here: main() already performed the
    # clean boundary, and calling it during retry can match the just-started
    # service wrapper through inherited command lines.
    services_log.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy(); env['PIPER_CAMERA_RESOLUTION']='960x540'; env['PIPER_SERVICE_LOG_DIR']=str(services_log)
    services_proc=subprocess.Popen([str(SERVICES)],env=env,start_new_session=True,stdout=(services_log/'services.log').open('a'),stderr=subprocess.STDOUT)
    # ros_preview_server.py blocks inside rospy.init_node while it retries a
    # missing master.  Wait for the service wrapper to bring up ROS first;
    # otherwise the web page reports a live preview process whose HTTP port
    # is not actually available.
    ros_ready = False
    # roslaunch may spend significant time scanning the >1 GB ROS log
    # directory before opening port 11311.  Keep the web process in startup
    # for up to a minute instead of declaring services down after 15 seconds.
    for _ in range(120):
        if services_proc is not None and services_proc.poll() is not None:
            break
        try:
            probe = subprocess.run(
                ['bash', '-lc', f'{ROS_SETUP}; rosnode list >/dev/null 2>&1'],
                timeout=1.5,
            )
            if probe.returncode == 0:
                ros_ready = True
                break
        except (OSError, subprocess.SubprocessError):
            pass
        time.sleep(0.5)
    # A previous web session may have left a preview process alive on 8766.
    # Replace it so stale ROS subscribers do not accumulate across retries.
    old_preview_pids = []
    for line in subprocess.run(['pgrep','-f',str(PREVIEW)], capture_output=True, text=True).stdout.splitlines():
        try:
            pid=int(line.strip()); old_preview_pids.append(pid); os.killpg(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    # Do not mistake a preview process that is still tearing down for a
    # healthy external server.  Otherwise the subsequent probe can succeed
    # in the short window before the old listener closes, leaving no preview
    # process after it exits.
    deadline = time.monotonic() + 5.0
    while old_preview_pids and time.monotonic() < deadline:
        old_preview_pids = [pid for pid in old_preview_pids if Path(f'/proc/{pid}').exists()]
        if old_preview_pids:
            time.sleep(0.1)
    preview_external = False
    preview_proc = None
    if not ros_ready:
        return
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
        if p.path=='/api/datasets':
            self._json({'datasets': list_datasets()}); return
        if p.path=='/api/status':
            preview_up = preview_external
            if not preview_up:
                try:
                    with socket.create_connection(('127.0.0.1', 8766), timeout=0.2):
                        preview_up = True
                except OSError:
                    preview_up = False
            state = capture_state()
            active = capture_proc is not None and capture_proc.poll() is None
            capture_status = 'running' if active and state == 'recording' else ('waiting' if active and state == 'waiting' else ('starting' if active else 'idle'))
            # Services may have been started from another terminal.  The ROS
            # master plus a live preview is still a valid service session;
            # only relying on the Popen handle would incorrectly show DOWN.
            services_up = (
                services_proc is not None and services_proc.poll() is None
            ) or (preview_up and ros_master_available())
            if capture_status == 'running':
                message = '采集中…'
            elif capture_status == 'waiting':
                message = '等待下一条 episode（按空格开始）'
            elif state == 'saving':
                message = '正在保存当前 episode…'
            elif capture_status == 'starting':
                message = '正在初始化同步采集…'
            elif not services_up:
                message = 'ROS服务未启动，请检查相机 USB 和运行日志'
            elif not preview_up:
                message = '相机预览未连接，请检查 ROS 相机话题'
            else:
                message = '服务就绪'
            next_episode = None
            if episode_file.exists():
                try: next_episode = int(episode_file.read_text(encoding='utf-8'))
                except ValueError: pass
            self._json({'running':capture_status == 'running','services':'up' if services_up else 'down','preview':'up' if preview_up else 'down','capture':capture_status,'message':message,'next_episode':next_episode,'log':tail_logs()}); return
        if p.path in ('/api/episode','/api/previews'):
            # Do not let the preview reader touch Parquet files while the
            # writer is updating them; pyarrow can crash on a partial file.
            if capture_proc is not None and capture_proc.poll() is None and capture_state() == 'recording':
                self._json({'meta':'当前 episode 采集中，完成后自动刷新','videos':[],'states':[],'items':[]}); return
            q=parse_qs(p.query); root=Path(q.get('dataset',[str(DEFAULT_DATASET)])[0]).expanduser().resolve(); ep=int(q.get('episode',[0])[0]); items=[]; states=[]
            wanted_names = {f'file-{ep:03d}.mp4', f'episode_{ep:06d}.mp4'}
            for f in sorted(root.glob('videos/**/*.mp4')):
                if f.name not in wanted_names:
                    continue
                items.append({'name':str(f.relative_to(root)),'url':'/media/'+quote(str(f.relative_to(root)))+'?dataset='+quote(str(root))})
                if len(items) >= 3:
                    break
            self._json({'meta':f'episode {ep} · {len(states)} 个关节采样 · {len(items)} 个视频','videos':items,'states':states,'items':items}); return
        if p.path == '/api/episodes':
            q = parse_qs(p.query); root = Path(q.get('dataset', [str(DEFAULT_DATASET)])[0]).expanduser().resolve()
            center = int(q.get('center', [0])[0]); all_rows = read_episode_rows(root); total = len(all_rows)
            if total:
                center_pos = next((i for i, row in enumerate(all_rows) if row['episode'] == center), total - 1)
                window_start = min(max(center_pos - 1, 0), max(total - 3, 0))
                rows = list(reversed(all_rows[window_start:window_start + min(3, total)]))
            else:
                rows = []
            self._json({'total': total, 'center': center, 'episodes': rows}); return
        if p.path.startswith('/media/'):
            q=parse_qs(p.query); root=Path(q.get('dataset',[str(DEFAULT_DATASET)])[0]).expanduser().resolve(); path=(root/unquote(p.path[7:])).resolve()
            if root not in path.parents or not path.is_file(): self.send_error(404); return
            data=path.read_bytes(); self.send_response(200); self.send_header('Content-Type','video/mp4'); self.send_header('Content-Length',str(len(data))); self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers cancel a media request when the episode selector
                # refreshes.  This is normal client behaviour, not a server
                # failure, and should not flood the runtime log with traces.
                pass
            return
        self.send_error(404)
    def do_POST(self):
        global capture_proc, capture_config
        if self.path=='/api/start':
            with capture_guard:
                if capture_proc is not None and capture_proc.poll() is None:
                    if capture_state() == 'waiting':
                        send_control('start'); self._json({'message':'下一条 episode 已开始'}); return
                    self._json({'error':'已有采集在运行'},409); return
                stale = active_collector_pids()
                if stale:
                    self._json({'error':f"检测到其他采集进程 PID {stale}，为避免损坏数据已拒绝启动"},409); return
                b=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
                info=Path(b['dataset_path']).expanduser().resolve()/'meta'/'info.json'
                if info.exists():
                    b['episode_idx']=int(json.loads(info.read_text(encoding='utf-8')).get('total_episodes',0))
                else:
                    b['episode_idx']=0
                if not topics_ready(timeout=30):
                    detail = topic_readiness_error or '三路相机或四路关节话题未发布'
                    self._json({'error':f'ROS 服务未就绪：{detail}，未启动采集器'},503)
                    return
                for path in (control_file, state_file, episode_file): path.unlink(missing_ok=True)
                args=[str(DIRECT),'--skip-can','--yes','--auto-start','--continuous']
                for k in ('dataset_path','repo_id','task','episode_idx','timesteps','fps','camera_resolution'): args += ['--'+k.replace('_','-'),str(b[k])]
                env=os.environ.copy(); env['PIPER_CONTROL_FILE']=str(control_file)
                # The web process owns ROS/camera/arm startup.  Prevent the
                # collector's legacy --auto-start path from launching a
                # second copy when topics were still coming online.
                env['PIPER_SKIP_SERVICE_AUTOSTART']='1'
                capture_config=b
                with capture_log.open('w') as out:
                    capture_proc=subprocess.Popen(args,cwd=ROOT,env=env,start_new_session=True,stdout=out,stderr=subprocess.STDOUT)
                self._json({'message':f"episode {b['episode_idx']} 已启动，正在等待同步帧",'episode_idx':b['episode_idx']}); return
        if self.path=='/api/stop':
            if capture_proc is not None and capture_proc.poll() is None: send_control('stop'); self._json({'message':'已请求停止当前 episode'})
            else: self._json({'message':'当前没有运行中的采集'}); return
        if self.path=='/api/shutdown':
            if capture_proc is not None and capture_proc.poll() is None:
                send_control('shutdown'); self._json({'message':'已请求结束采集会话，当前 episode 将先保存'})
            else:
                self._json({'message':'当前没有运行中的采集'})
            return
        if self.path=='/api/retry':
            with capture_guard:
                if capture_proc is not None and capture_proc.poll() is None:
                    self._json({'error':'采集运行中，不能重启 ROS 服务，请先结束采集'},409); return
                self._json({'error':'请在终端按 Ctrl-C 后重新运行启动命令；监督脚本会先完整清理再启动新服务'},409); return
        self.send_error(404)

def main():
    # Keep process supervision in Bash.  It can reliably trap Ctrl-C, clean
    # old ROS process groups, start fresh services, and then run this HTTP
    # server.  Preserve the familiar Python command by transparently
    # replacing it with the supervisor on first entry.
    if os.environ.get('PIPER_COLLECT_SUPERVISED') != '1':
        os.execv(str(RUNNER), [str(RUNNER)])
    acquire_web_lock()
    # Register cleanup only for the actual web process.  Importing this
    # module for diagnostics/tests must not terminate an unrelated running
    # ROS service session at interpreter exit.
    atexit.register(cleanup_runtime)
    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)
    # Clear every process group and temporary IPC artifact from an earlier
    # crash before starting ROS, cameras, arms, preview, or a collector.
    port=int(os.environ.get('PIPER_COLLECT_WEB_PORT','8765'))
    server = ThreadingHTTPServer(('0.0.0.0',port),Handler)
    server.daemon_threads = True
    print(f'http://127.0.0.1:{port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n正在保存当前 episode 并关闭采集服务……', flush=True)
    finally:
        server.server_close()
        cleanup_runtime()

if __name__=='__main__':
    main()
