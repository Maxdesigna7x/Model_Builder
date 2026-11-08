"""Persistent local dataset curation, independent from training state."""
from __future__ import annotations
import base64, hashlib, io, json, math, os, random, re, shutil, subprocess, sys, tempfile, time, uuid
from contextlib import contextmanager
from pathlib import Path
from PIL import Image, ImageOps
from .storage import atomic_json, project_dir, iso_now
from .errors import BackendError

EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
def uid(): return uuid.uuid4().hex

def root_of(project):
    root=project_dir(project)
    if not (root/'project.json').exists(): raise BackendError('PROJECT_MISSING_MANIFEST')
    return root

@contextmanager
def locked(root):
    """OS-managed lock: released on process death, never expires while held."""
    with open(root/'.dataset.lock','a+b') as handle:
        if os.name=='nt':
            import msvcrt
            handle.seek(0,os.SEEK_END)
            if handle.tell()==0: handle.write(b'0'); handle.flush()
            def acquire():
                handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            def release():
                handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            def acquire(): fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            def release(): fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        for _ in range(400):
            try: acquire(); break
            except OSError: time.sleep(.05)
        else: raise BackendError('DATASET_BUSY','Try again')
        try: yield
        finally: release()

def load(root):
    p=root/'dataset-project.json'
    if not p.exists(): return {'schema_version':1,'sources':[],'samples':[],'classes':[],'groups':[],'jobs':[],'exports':[]}
    data=json.loads(p.read_text(encoding='utf8'))
    if data.get('schema_version')!=1: raise BackendError('DATASET_SCHEMA_UNSUPPORTED')
    return data

def save(root,data): atomic_json(root/'dataset-project.json',data)
def mutate(root,fn,revision_ids=()):
    with locked(root):
        data=load(root)
        before={s['id']:json.dumps(s,sort_keys=True,ensure_ascii=False) for s in data['samples']}
        result=fn(data)
        changed=set()
        for item in data['samples']:
            old=before.get(item['id'])
            if old is None: item.setdefault('_revision',1); changed.add(item['id'])
            elif json.dumps(item,sort_keys=True,ensure_ascii=False)!=old:
                item['_revision']=item.get('_revision',0)+1; changed.add(item['id'])
        if isinstance(result,dict) and result.get('saved'):
            requested=changed|set(revision_ids)
            result['revisions']={s['id']:s.get('_revision',0) for s in data['samples'] if s['id'] in requested}
        save(root,data); return result

def image_path(root,sample):
    path=Path(sample['path'])
    if not path.is_absolute():
        path=(root/path).resolve()
        if not path.is_relative_to(root.resolve()): raise BackendError('DATASET_INVALID_PATH')
    if not path.is_file(): raise BackendError('DATASET_SOURCE_MISSING',str(path))
    return path

def read(project,page=0,page_size=48,filters=None):
    root=root_of(project); data=load(root)
    dead=[]
    for job in data['jobs']:
        if job.get('state') in ('pending','running'):
            pid=job.get('pid'); gone=not pid
            if pid:
                try:
                    os.kill(int(pid),0)
                    stat=Path(f'/proc/{int(pid)}/stat')
                    if stat.is_file() and stat.read_text().rsplit(') ',1)[-1].startswith('Z '): gone=True
                except ProcessLookupError: gone=True
                except (PermissionError,OSError): pass
            if gone: dead.append((job['id'],pid,job.get('run_token')))
    if dead:
        def recover(d):
            for job in d['jobs']:
                if (job['id'],job.get('pid'),job.get('run_token')) in dead and job['state'] in ('pending','running'): job.update(state='interrupted',error='Worker interrupted; resume to continue')
        mutate(root,recover); data=load(root)
    samples=data['samples']; filters=filters or {}
    for key in ('source_id','status','group_id','class_id'):
        if filters.get(key): samples=[s for s in samples if (filters[key] in s.get('class_ids',[]) if key=='class_id' else s.get(key)==filters[key])]
    all_samples=data['samples']; hashes={}; counts={}; class_counts={c['id']:0 for c in data['classes']}; group_counts={}; small=[]; min_size=max(1,int(filters.get('min_size',32)))
    for item in all_samples:
        hashes.setdefault(item.get('sha256'),[]).append(item['id'])
        state=item.get('status','unreviewed'); counts[state]=counts.get(state,0)+1
        for c in item.get('class_ids',[]): class_counts[c]=class_counts.get(c,0)+1
        group=item.get('group_id'); group_counts[group]=group_counts.get(group,0)+1
        if item.get('width',0)<min_size or item.get('height',0)<min_size: small.append(item['id'])
    summary={'counts':counts,'class_counts':class_counts,'group_counts':{str(k or 'unassigned'):v for k,v in group_counts.items()},'unassigned':sum(not s.get('class_ids') for s in all_samples),'duplicates':[v for k,v in hashes.items() if k and len(v)>1], 'small':small,'read_errors':sum(len(src.get('errors',[])) for src in data['sources'])}
    alert=filters.get('alert')
    if alert=='small': samples=[s for s in samples if s['id'] in set(small)]
    elif alert=='duplicate':
        duplicate_ids={sid for group in summary['duplicates'] for sid in group}; samples=[s for s in samples if s['id'] in duplicate_ids]
    elif alert=='unassigned': samples=[s for s in samples if not s.get('class_ids')]
    elif alert: raise BackendError('DATASET_FILTER_INVALID')
    page_size=max(1,min(100,int(page_size))); page=max(0,int(page)); total=len(samples)
    exports=[{**{k:v for k,v in e.items() if k!='samples'},'sample_count':len(e.get('samples',[]))} for e in data['exports']]
    jobs=[{k:v for k,v in job.items() if k!='sample_ids'} for job in data['jobs']]
    return {**data,'exports':exports,'jobs':jobs,'samples':samples[page*page_size:(page+1)*page_size], 'total':total,'page':page,'page_size':page_size,'summary':summary}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def image(project,sample_id,thumbnail=True):
    root=root_of(project); s=next((x for x in load(root)['samples'] if x['id']==sample_id),None)
    if not s: raise BackendError('DATASET_SAMPLE_NOT_FOUND')
    path=image_path(root,s)
    if s.get('sha256') and sha(path)!=s['sha256']: raise BackendError('DATASET_SOURCE_CHANGED',str(path))
    cache=root/'thumbnails'/f"{s['id']}.jpg"
    if thumbnail and cache.is_file(): content=cache.read_bytes()
    else:
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            if thumbnail: im.thumbnail((240,240))
            out=io.BytesIO(); im.save(out,'JPEG',quality=85); content=out.getvalue()
        if thumbnail:
            cache.parent.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache.parent,delete=False) as out: out.write(content); temporary=Path(out.name)
            temporary.replace(cache)
    return {'url':'data:image/jpeg;base64,'+base64.b64encode(content).decode()}

def sample(path,source_id,**extra):
    digest=sha(path)
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im); im.load(); width,height=im.size
    return {'id':uid(),'path':str(path),'source_id':source_id,'width':width,'height':height,'sha256':digest,'class_ids':[],'proposals':[],'boxes':[],'status':'unreviewed','included':True,**extra}

def update(project,changes):
    root=root_of(project)
    def apply(d):
        for item in changes.get('classes',[]):
            name=str(item.get('name','')).strip()
            if not name: raise BackendError('DATASET_CLASS_NAME_REQUIRED')
            found=next((c for c in d['classes'] if c['id']==item.get('id')),None)
            if found: found.update(name=name,description=str(item.get('description','')))
            else: d['classes'].append({'id':uid(),'name':name,'description':str(item.get('description',''))})
        class_ids={c['id'] for c in d['classes']}
        for patch in changes.get('samples',[]):
            s=next((s for s in d['samples'] if s['id']==patch.get('id')),None)
            if not s: raise BackendError('DATASET_SAMPLE_NOT_FOUND')
            if '_revision' in patch and patch['_revision']!=s.get('_revision',0): raise BackendError('DATASET_SAMPLE_CONFLICT','Sample changed; reload before saving')
            if 'class_ids' in patch and not set(patch['class_ids'])<=class_ids: raise BackendError('DATASET_CLASS_NOT_FOUND')
            if 'status' in patch and patch['status'] not in ('unreviewed','annotated','negative','excluded'): raise BackendError('DATASET_STATUS_INVALID')
            if 'boxes' in patch:
                for b in patch['boxes']:
                    if b.get('class_id') not in class_ids: raise BackendError('DATASET_CLASS_NOT_FOUND')
                    xy=b.get('xywh',[])
                    if len(xy)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in xy): raise BackendError('DATASET_BOX_INVALID')
                    x,y,w,h=xy
                    if w<=0 or h<=0 or x-w/2 < -1e-9 or y-h/2 < -1e-9 or x+w/2 > 1+1e-9 or y+h/2 > 1+1e-9: raise BackendError('DATASET_BOX_INVALID')
                    b.setdefault('id',uid()); b.setdefault('origin','human'); b.setdefault('accepted',True)
            for key in ('class_ids','boxes','status','included','group_id','proposals'):
                if key in patch: s[key]=patch[key]
        for patch in changes.get('groups',[]):
            g=next((g for g in d['groups'] if g['id']==patch.get('id')),None)
            if g: g['name']=str(patch.get('name',g['name']))
            else: d['groups'].append({'id':uid(),'name':str(patch.get('name','Grupo'))})
        selected=changes.get('accept_proposals')
        if selected is not None:
            samples={s['id']:s for s in d['samples']}; selected=set(selected)
            if not selected<=samples.keys(): raise BackendError('DATASET_SAMPLE_NOT_FOUND')
            for sid in selected:
                item=samples[sid]
                if item.get('class_ids') or not item.get('included',True) or item.get('status') in ('excluded','negative'): continue
                proposals=[p for p in item.get('proposals',[]) if p.get('class_id') in class_ids and (p.get('eligible') is True or (p.get('origin')=='yolo' and 'eligible' not in p))]
                if len(proposals)==1:
                    item['class_ids']=[proposals[0]['class_id']]; item['status']='annotated'
        accepted_groups=changes.get('accept_group_proposals')
        if accepted_groups is not None:
            samples={s['id']:s for s in d['samples']}; selected=set(accepted_groups); groups={g['id'] for g in d['groups']}
            if not selected<=samples.keys(): raise BackendError('DATASET_SAMPLE_NOT_FOUND')
            for sid in selected:
                item=samples[sid]; proposed=item.get('group_proposal')
                if not item.get('group_id') and proposed in groups:
                    item['group_id']=proposed; item.pop('group_proposal',None)
        discard=changes.get('discard_proposals')
        if discard is not None:
            samples={s['id']:s for s in d['samples']}; selected=set(discard.get('ids',[])); kind=discard.get('kind')
            if not selected<=samples.keys(): raise BackendError('DATASET_SAMPLE_NOT_FOUND')
            if kind not in ('class','group','all'): raise BackendError('DATASET_PROPOSAL_KIND_INVALID')
            for sid in selected:
                if kind in ('class','all'): samples[sid]['proposals']=[]
                if kind in ('group','all'): samples[sid].pop('group_proposal',None)
        merge=changes.get('merge_groups')
        if merge:
            source,target=merge.get('source'),merge.get('target')
            group_ids={g['id'] for g in d['groups']}
            if source==target or source not in group_ids or target not in group_ids: raise BackendError('DATASET_GROUP_INVALID')
            for item in d['samples']:
                if item.get('group_id')==source: item['group_id']=target
                if item.get('group_proposal')==source: item['group_proposal']=target
            d['groups']=[g for g in d['groups'] if g['id']!=source]
        separate=changes.get('separate_samples')
        if separate:
            selected=set(separate.get('ids',[])); samples={s['id']:s for s in d['samples']}
            if not selected or not selected<=samples.keys(): raise BackendError('DATASET_SAMPLE_NOT_FOUND')
            group={'id':uid(),'name':str(separate.get('name') or f"Grupo {len(d['groups'])+1}")}; d['groups'].append(group)
            for sid in selected: samples[sid]['group_id']=group['id']
        return {'saved':True,'classes':d['classes'],'groups':d['groups']}
    requested={item['id'] for item in changes.get('samples',[]) if 'id' in item}
    requested.update(changes.get('accept_proposals') or [])
    requested.update(changes.get('accept_group_proposals') or [])
    requested.update((changes.get('discard_proposals') or {}).get('ids',[]))
    requested.update((changes.get('separate_samples') or {}).get('ids',[]))
    return mutate(root,apply,requested)

def estimate(path,options):
    import cv2
    cap=cv2.VideoCapture(str(Path(path).expanduser()))
    if not cap.isOpened(): raise BackendError('DATASET_VIDEO_OPEN_FAILED',str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); start=float(options.get('start',0)); end=options.get('end'); mode=options.get('mode','frames'); interval=float(options.get('interval',1))
        if not math.isfinite(start) or not math.isfinite(interval) or start<0 or (end is not None and (not math.isfinite(float(end)) or float(end)<start)) or interval<=0 or (mode=='frames' and interval!=int(interval)) or mode not in ('frames','seconds') or (options.get('max_frames') is not None and (int(options['max_frames'])<1 or float(options['max_frames'])!=int(options['max_frames']))): raise BackendError('DATASET_VIDEO_OPTIONS_INVALID')
        if (mode=='seconds' or start or end is not None) and (not math.isfinite(fps) or fps<=0): raise BackendError('DATASET_VIDEO_FPS_UNKNOWN','Temporal sampling requires a reliable FPS')
        first=math.ceil(start*fps) if start else 0; last=min(count,math.floor(float(end)*fps)+1) if end is not None and count>0 else count
        n=max(0,math.ceil((last-first)/(interval if mode=='frames' else max(1,interval*fps)))) if count>0 else None
        if n is not None and options.get('max_frames'): n=min(n,int(options['max_frames']))
        return {'fps':fps if fps>0 and math.isfinite(fps) else None,'frames':count or None,'estimated':n,'rounding':'ceil(start*fps); temporal targets use first frame at or after each target'}
    finally: cap.release()

def launch(root,job):
    token=uid(); job.update(run_token=token,state='pending',cancel_requested=False)
    try:
        child=subprocess.Popen([sys.executable,'-m','modelbuilder.dataset_project','worker',str(root),job['id'],token],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')})
        job['pid']=child.pid
    except Exception as exc:
        job.update(state='error',error='Cannot start worker: '+str(exc),finished_at=iso_now())

def start(project,operation,options):
    root=root_of(project)
    if operation not in ('images','video','yolo','yolo_info','embeddings','export'): raise BackendError('DATASET_JOB_UNSUPPORTED',str(operation))
    job={'id':uid(),'operation':operation,'options':options,'state':'pending','processed':0,'total':None,'cursor':0,'errors':[],'created_at':iso_now()}
    def add(d):
        d['jobs'].append(job); launch(root,job)
        return job
    return mutate(root,add)

def control(project,job_id,action):
    root=root_of(project)
    def change(d):
        j=next((j for j in d['jobs'] if j['id']==job_id),None)
        if not j: raise BackendError('DATASET_JOB_NOT_FOUND')
        if action=='cancel' and j['state'] in ('pending','running'): j['cancel_requested']=True
        elif action=='resume' and j['state'] in ('cancelled','error','interrupted'):
            j.pop('error',None); j.pop('finished_at',None); launch(root,j)
        else: raise BackendError('DATASET_JOB_STATE_INVALID')
        return j
    return mutate(root,change)

def worker(root,job_id,run_token=None):
    root=Path(root)
    def job_update(fn):
        def apply(d):
            j=next(j for j in d['jobs'] if j['id']==job_id); return fn(d,j)
        return mutate(root,apply)
    def begin(d,j):
        if j.get('state')!='pending' or j.get('run_token')!=run_token: return None
        if j.get('cancel_requested'): j.update(state='cancelled',finished_at=iso_now()); return None
        j.update(state='running',pid=os.getpid()); return dict(j)
    j=job_update(begin)
    if j is None: return
    opt=j['options']
    try:
        if j['operation']=='images':
            path=Path(opt['path']).expanduser().resolve()
            if not path.is_dir(): raise ValueError(f'Image directory not found: {path}')
            inventory=root/'jobs'/f'{job_id}-images.json'
            if inventory.is_file(): files=[Path(p) for p in json.loads(inventory.read_text(encoding='utf8'))]
            else:
                files=sorted(p for p in (path.rglob('*') if opt.get('recursive',True) else path.glob('*')) if p.is_file() and p.suffix.lower() in EXTENSIONS)
                atomic_json(inventory,[str(p) for p in files])
            def init(d,j):
                j['total']=len(files)
                if not j.get('source_id'):
                    j['source_id']=uid(); d['sources'].append({'id':j['source_id'],'type':'images','path':str(path),'count':0,'errors':[]})
                return j['source_id']
            sid=job_update(init)
            for i,p in enumerate(files):
                if i<j['cursor']: continue
                def one(d,job):
                    if job.get('cancel_requested'): return False
                    src=next(s for s in d['sources'] if s['id']==sid)
                    try:
                        if not any(s['source_id']==sid and s['path']==str(p) for s in d['samples']):
                            s=sample(p,sid)
                            if opt.get('labels') and p.parent!=path:
                                name=p.parent.name; c=next((c for c in d['classes'] if c['name']==name),None)
                                if c is None: c={'id':uid(),'name':name,'description':''}; d['classes'].append(c)
                                s['class_ids']=[c['id']]; s['status']='annotated'
                            d['samples'].append(s); src['count']+=1
                    except Exception as e: src['errors'].append({'path':str(p),'error':str(e)}); job['errors'].append({'path':str(p),'error':str(e)})
                    job.update(cursor=i+1,processed=i+1); return True
                if not job_update(one): break
        elif j['operation']=='video': video_worker(root,j,job_update)
        elif j['operation']=='export': export_worker(root,j,job_update)
        else:
            from . import dataset_models
            dataset_models.run(root,j,job_update)
        def finish(d,j): j.update(state='cancelled' if j.get('cancel_requested') else 'completed',finished_at=iso_now())
        job_update(finish)
    except Exception as e:
        def fail(d,j): j.update(state='error',error=str(e),finished_at=iso_now())
        job_update(fail)

def video_worker(root,j,job_update):
    import cv2
    opt=j['options']; path=Path(opt['path']).expanduser().resolve(); info=estimate(str(path),opt); fps=info['fps']; mode=opt.get('mode','frames'); interval=float(opt.get('interval',1)); first=math.ceil(float(opt.get('start',0))*fps) if opt.get('start') else 0; end=float(opt['end']) if opt.get('end') is not None else None; limit=int(opt.get('max_frames') or 0)
    fingerprint=sha(path); signature=hashlib.sha256(json.dumps({'path':str(path),'sha':fingerprint,'mode':mode,'interval':interval,'start':opt.get('start',0),'end':end,'limit':limit},sort_keys=True).encode()).hexdigest()
    def init(d,job):
        src=next((s for s in d['sources'] if s.get('signature')==signature),None)
        if not src: src={'id':uid(),'type':'video','path':str(path),'sha256':fingerprint,'signature':signature,'count':0,'errors':[]}; d['sources'].append(src)
        job.update(source_id=src['id'],total=info['estimated']); return src['id']
    sid=job_update(init); cap=cv2.VideoCapture(str(path)); idx=-1; selected=0; previous_ts=None; first_timestamp=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            idx+=1
            pos=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            ts=pos if math.isfinite(pos) and pos>=0 else None
            if mode=='seconds':
                if ts is None or (previous_ts is not None and ts<=previous_ts): raise ValueError('Decoder did not provide reliable increasing timestamps; use frame sampling')
                previous_ts=ts
                if ts<float(opt.get('start',0)): continue
                if end is not None and ts>end: break
                if first_timestamp is None: first_timestamp=ts
                choose=ts+1e-9>=first_timestamp+selected*interval
            else:
                if idx<first: continue
                if end is not None and idx/fps>end: break
                choose=(idx-first)%int(interval)==0
            if not choose: continue
            if limit and selected>=limit: break
            selected+=1
            if selected<=j['cursor']: continue
            relative=Path('media')/sid/f'{idx:012d}.jpg'; out=root/relative; out.parent.mkdir(parents=True,exist_ok=True)
            def one(d,job):
                if job.get('cancel_requested'): return False
                src=next(s for s in d['sources'] if s['id']==sid)
                if not any(s['source_id']==sid and s.get('frame_index')==idx for s in d['samples']):
                    if not cv2.imwrite(str(out),frame): raise ValueError(f'Cannot write frame {idx}')
                    s=sample(out,sid,frame_index=idx,timestamp=ts if ts is not None else None,video_path=str(path)); s['path']=relative.as_posix(); d['samples'].append(s); src['count']+=1
                job.update(cursor=selected,processed=selected); return True
            if not job_update(one): break
    finally: cap.release()

def export(project,options,preview=False,*,_data=None,_job_id=None,_progress=None,_commit=None):
    root=root_of(project); fmt=options.get('format'); seed=int(options.get('seed',42)); parts=('train','val','test')
    ratios=[float(options.get('train',70)),float(options.get('validation',15)),float(options.get('test',15))]
    if fmt not in ('classification','detection') or any(x<0 for x in ratios) or not math.isclose(sum(ratios),100): raise BackendError('DATASET_EXPORT_OPTIONS_INVALID')
    data=_data if _data is not None else load(root); classes=data['classes']; ids={c['id']:i for i,c in enumerate(classes)}
    if not classes: raise BackendError('DATASET_CLASS_NAME_REQUIRED')
    names=[re.sub(r'[^\w.-]+','_',c['name'],flags=re.UNICODE).strip('._') or 'class' for c in classes]
    if len(set(x.casefold() for x in names))!=len(names): raise BackendError('DATASET_CLASS_NAME_COLLISION','Class names must be unique after sanitization')
    def valid_box(b):
        xy=b.get('xywh',[])
        if b.get('class_id') not in ids or not isinstance(xy,(list,tuple)) or len(xy)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in xy): return False
        x,y,w,h=xy
        return w>0 and h>0 and x-w/2>=-1e-9 and y-h/2>=-1e-9 and x+w/2<=1+1e-9 and y+h/2<=1+1e-9
    eligible=[]; omitted={}
    for s in data['samples']:
        reason=None
        if not s.get('included',True) or s.get('status')=='excluded': reason='excluded'
        elif fmt=='classification':
            if len(s.get('class_ids',[]))>1: raise BackendError('DATASET_AMBIGUOUS_CLASS',s['id'])
            if len(s.get('class_ids',[]))!=1 or s['class_ids'][0] not in ids: reason='unassigned'
        elif s.get('status')=='negative':
            if any(b.get('accepted') is True for b in s.get('boxes',[])): reason='negative_with_boxes'
        elif s.get('status')!='annotated' or not any(b.get('accepted') is True for b in s.get('boxes',[])): reason='unreviewed_or_empty'
        elif any(not valid_box(b) for b in s.get('boxes',[]) if b.get('accepted') is True): reason='invalid_box'
        if reason: omitted[reason]=omitted.get(reason,0)+1
        else: eligible.append(s)
    if not eligible:
        if preview: return {'counts':dict.fromkeys(parts,0),'omitted':omitted,'source_groups':0,'warnings':['No eligible samples.']}
        raise BackendError('DATASET_EXPORT_EMPTY',json.dumps(omitted))
    parent={s['id']:s['id'] for s in eligible}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b): parent[find(a)]=find(b)
    keys={}; byid={s['id']:s for s in data['samples']}; sources={s['id']:s for s in data['sources']}
    for s in eligible:
        ancestor=s; seen=set()
        while ancestor.get('parent_sample_id') in byid and ancestor['id'] not in seen:
            seen.add(ancestor['id']); ancestor=byid[ancestor['parent_sample_id']]
        src=sources.get(ancestor['source_id'],{})
        group=('video',src.get('sha256') or src.get('path')) if src.get('type')=='video' else ('sample',ancestor['id'])
        for key in (group,('sha',s['sha256'])):
            if key in keys: union(s['id'],keys[key])
            else: keys[key]=s['id']
    groups={}
    for s in eligible: groups.setdefault(find(s['id']),[]).append(s)
    blocks=list(groups.values()); random.Random(seed).shuffle(blocks); blocks.sort(key=len,reverse=True)
    totals={p:0 for p in parts}; assignments={}
    for block in blocks:
        choices=[i for i,r in enumerate(ratios) if r>0]
        i=max(choices,key=lambda k:(len(eligible)*ratios[k]/100-totals[parts[k]],-k)); p=parts[i]; totals[p]+=len(block)
        for s in block: assignments[s['id']]=p
    warnings=[]
    if len(groups)<3: warnings.append('Fewer than three source groups; nonempty partitions cannot be guaranteed.')
    if any(abs(totals[p]/len(eligible)*100-ratios[i])>5 for i,p in enumerate(parts)): warnings.append('Actual split percentages differ from requested percentages because source groups remain together.')
    if preview: return {'counts':totals,'omitted':omitted,'source_groups':len(groups),'warnings':warnings}
    name=re.sub(r'[^\w.-]+','-',str(options.get('name','dataset')),flags=re.UNICODE).strip('.-') or 'dataset'
    final=root/'exports'/f'{name}-{_job_id if _job_id else uid()[:8]}'
    out=(root/'exports'/f'.pending-{_job_id}') if _job_id else final
    if _job_id and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest=[]
    try:
        if _progress and not _progress(0,len(eligible)): raise ExportCancelled()
        if fmt=='detection':
            (out/'classes.txt').write_text('\n'.join(names)+'\n',encoding='utf8')
            (out/'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {json.dumps(n,ensure_ascii=False)}\n' for i,n in enumerate(names)),encoding='utf8')
        for s in eligible:
            source=image_path(root,s)
            if sha(source)!=s['sha256']: raise BackendError('DATASET_SOURCE_CHANGED',str(source))
            split=assignments[s['id']]; filename=s['id']+'.jpg'
            target=out/(Path(split) if fmt=='classification' else Path('images')/split)
            if fmt=='classification': target=target/names[ids[s['class_ids'][0]]]
            target.mkdir(parents=True,exist_ok=True)
            with Image.open(source) as im: ImageOps.exif_transpose(im).convert('RGB').save(target/filename,'JPEG',quality=95)
            if fmt=='detection':
                labeldir=out/'labels'/split; labeldir.mkdir(parents=True,exist_ok=True)
                lines=[]
                for b in s.get('boxes',[]):
                    if b.get('accepted') is True: lines.append(str(ids[b['class_id']])+' '+' '.join(f'{float(v):.8f}' for v in b['xywh']))
                (labeldir/(s['id']+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf8')
            manifest.append({'sample_id':s['id'],'source_id':s['source_id'],'parent_sample_id':s.get('parent_sample_id'),'sha256':s['sha256'],'split':split,'file':str((target/filename).relative_to(out))})
            if _progress and not _progress(len(manifest),len(eligible)): raise ExportCancelled()
        result={'id':_job_id or uid(),'created_at':iso_now(),'path':str(final),'format':fmt,'options':options,'classes':classes,'samples':manifest,'counts':totals,'omitted':omitted,'source_groups':len(groups),'warnings':warnings}
        atomic_json(out/'manifest.json',result)
        if _commit:
            if not _commit(result,out,final): raise ExportCancelled()
        else: mutate(root,lambda d:d['exports'].append(result))
        return result
    except Exception:
        shutil.rmtree(out,ignore_errors=True); raise

class ExportCancelled(Exception):
    pass

def export_worker(root,job,job_update):
    """Restart from an immutable manifest snapshot and publish a completed directory once."""
    job_id=job['id']; options=job['options']
    name=re.sub(r'[^\w.-]+','-',str(options.get('name','dataset')),flags=re.UNICODE).strip('.-') or 'dataset'
    final=root/'exports'/f'{name}-{job_id}'; stage=root/'exports'/f'.pending-{job_id}'
    def register(data,j,result):
        if not any(item.get('id')==job_id for item in data['exports']): data['exports'].append(result)
        j['result']={k:v for k,v in result.items() if k!='samples'}
        j['phase']='complete'; j['processed']=len(result['samples']); j['total']=len(result['samples'])
    if final.exists():
        path=final/'manifest.json'
        if not path.is_file(): raise ValueError(f'Export destination already exists: {final}')
        result=json.loads(path.read_text(encoding='utf8'))
        if result.get('id')!=job_id: raise ValueError(f'Export destination already exists: {final}')
        job_update(lambda d,j:register(d,j,result))
        if stage.exists(): shutil.rmtree(stage)
        return
    snapshot=root/'jobs'/f'{job_id}-export.json'
    if not snapshot.exists():
        current=load(root)
        atomic_json(snapshot,{key:current[key] for key in ('schema_version','sources','samples','classes','groups')})
    data=json.loads(snapshot.read_text(encoding='utf8'))
    project=json.loads((root/'project.json').read_text(encoding='utf8')); project['path']=str(root)
    def progress(done,total):
        def apply(d,j):
            if j.get('cancel_requested'): return False
            j.update(processed=done,total=total,cursor=done,phase='exporting'); return True
        return job_update(apply)
    def commit(result,out,destination):
        def apply(d,j):
            if j.get('cancel_requested'): return False
            out.rename(destination)
            register(d,j,result)
            return True
        return job_update(apply)
    try: export(project,options,_data=data,_job_id=job_id,_progress=progress,_commit=commit)
    except ExportCancelled: pass

if __name__=='__main__' and len(sys.argv)==5 and sys.argv[1]=='worker': worker(sys.argv[2],sys.argv[3],sys.argv[4])
