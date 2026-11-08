"""Optional model-backed dataset operations, imported only on demand."""
import os
import hashlib
import math
from PIL import Image, ImageOps
from .dataset_project import image_path, sample, sha, uid

YOLO = {'YOLO26':['n','s','m','l','x'],'YOLO12':['n','s','m','l','x'],'YOLO11':['n','s','m','l','x'],'YOLOv10':['n','s','m','b','l','x'],'YOLOv9':['t','s','m','c','e'],'YOLOv8':['n','s','m','l','x'],'YOLOv5u':['n','s','m','l','x']}
def checkpoint(family,size):
    if family not in YOLO or size not in YOLO[family]: raise ValueError('Unsupported YOLO family and size')
    prefix={'YOLO26':'yolo26','YOLO12':'yolo12','YOLO11':'yolo11','YOLOv10':'yolov10','YOLOv9':'yolov9','YOLOv8':'yolov8','YOLOv5u':'yolov5'}[family]
    return prefix+size+('u' if family=='YOLOv5u' else '')+'.pt'

def run(root,job,job_update):
    if job['operation']=='embeddings': return run_embeddings(root,job,job_update)
    if job['operation'] not in ('yolo','yolo_info'): raise ValueError('Unsupported model operation')
    from ultralytics import YOLO as Detector
    options=job['options']; name=checkpoint(options.get('family','YOLO11'),options.get('size','n'))
    conf=float(options.get('confidence',.25)); padding=int(options.get('padding',0)); minimum=int(options.get('min_size',0)); action=options.get('action','boxes')
    if not 0 <= conf <= 1 or padding < 0 or minimum < 0 or action not in ('boxes','crops','both'): raise ValueError('Invalid detection options')
    if job_update(lambda data,j: bool(j.get('cancel_requested'))): return
    cache=root/"cache"/"yolo"; cache.mkdir(parents=True,exist_ok=True)
    old_cwd=os.getcwd()
    try:
        os.chdir(cache)
        model=Detector(name)
    finally:
        os.chdir(old_cwd)
    if job['operation']=='yolo_info':
        def info(data,j):
            if j.get('cancel_requested'): return
            j['classes']=[str(model.names[k]) for k in sorted(model.names)]
            j['checkpoint']=name; j['processed']=1; j['total']=1
        job_update(info)
        return
    selected=set(options.get("classes") or [])
    available=set(map(str,model.names.values()))
    if selected-available: raise ValueError("Unknown detector classes: "+", ".join(sorted(selected-available)))
    def init(data,j):
        if 'sample_ids' not in j: j['sample_ids']=[s['id'] for s in data['samples'] if s.get('included',True)]
        j['total']=len(j['sample_ids']); return list(j['sample_ids'])
    ids=job_update(init)
    for index,sid in enumerate(ids):
        if index<job.get('cursor',0): continue
        def get(data,j):
            if j.get('cancel_requested'): return None
            return next((dict(s) for s in data['samples'] if s['id']==sid),None)
        s=job_update(get)
        if s is None: break
        crop_paths=[]
        try:
            path=image_path(root,s)
            if sha(path)!=s.get('sha256'): raise ValueError(f'Source changed: {path}')
            with Image.open(path) as img: original=ImageOps.exif_transpose(img).convert('RGB')
            prediction=model.predict(original,conf=conf,device='cpu',verbose=False)[0]
            if job_update(lambda data,j: bool(j.get('cancel_requested'))): return
            detections=[]; selected=set(options.get('classes') or [])
            for b in prediction.boxes:
                cls=int(b.cls.item()); label=str(prediction.names[cls]); score=float(b.conf.item())
                if selected and label not in selected: continue
                x1,y1,x2,y2=map(float,b.xyxy[0].tolist()); w,h=original.size
                if not all(math.isfinite(v) for v in (x1,y1,x2,y2,score)) or not 0<=score<=1: raise ValueError('Detector returned invalid coordinates or confidence')
                x1=max(0,min(w,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
                if x2<=x1 or y2<=y1 or x2-x1<minimum or y2-y1<minimum: continue
                crop_path=None
                if action in ('crops','both'):
                    key=hashlib.sha256(f"{job['id']}:{sid}:{len(detections)}".encode()).hexdigest()
                    crop_path=root/'media'/'crops'/f'{key}.jpg'; crop_path.parent.mkdir(parents=True,exist_ok=True); crop_paths.append(crop_path)
                    original.crop((max(0,int(x1)-padding),max(0,int(y1)-padding),min(w,int(x2+0.999)+padding),min(h,int(y2+0.999)+padding))).save(crop_path,quality=95)
                detections.append((label,score,[(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h],crop_path))
            def apply(data,j):
                if j.get('cancel_requested'): return False
                target=next(x for x in data['samples'] if x['id']==sid)
                for label,score,xy,crop_path in detections:
                    c=next((c for c in data['classes'] if c['name']==label),None)
                    if not c: c={'id':uid(),'name':label,'description':''};data['classes'].append(c)
                    if action in ('boxes','both'): target['boxes'].append({'id':uid(),'class_id':c['id'],'xywh':xy,'origin':'yolo','accepted':False,'model':name,'confidence':score,'job_id':j['id']})
                    if crop_path: data['samples'].append(sample(crop_path,s['source_id'],parent_sample_id=sid,class_ids=[],proposals=[{'class_id':c['id'],'similarity':score,'origin':'yolo'}],crop_box=xy,model=name))
                j.update(cursor=index+1,processed=index+1); return True
            if not job_update(apply):
                for path in crop_paths: path.unlink(missing_ok=True)
                return
        except Exception as exc:
            for path in crop_paths: path.unlink(missing_ok=True)
            def failed(data,j): j['errors'].append({'sample_id':sid,'error':str(exc)});j.update(cursor=index+1,processed=index+1)
            job_update(failed)

# Pinned model revision and preprocessing fingerprint are part of each cache key.
EG2_MODEL = 'google/embeddinggemma-2'
EG2_REVISION = 'c3f2237ce16fe33d1bb9dab8692c29c79865bdeb'
EG2_PREPROCESS = 'exif-rgb-transformers-default-meanpool-v1'

def run_embeddings(root, job, job_update):
    """Compute real EG2 vectors once per source image, then cluster or classify."""
    import hashlib, json, os, tempfile
    import numpy as np
    from .dataset_project import sha
    if job.get('phase')=='complete': return
    opt=job['options']; mode=opt.get('mode','cluster')
    if mode not in ('cluster','classify'): raise ValueError('Choose cluster or classify')
    threshold=float(opt.get('threshold',0.0)); margin=float(opt.get('margin',0.05))
    if not -1<=threshold<=1 or not 0<=margin<=2: raise ValueError('Invalid similarity threshold or margin')
    def init(data,j):
        if 'sample_ids' not in j:
            selected=set(opt.get('sample_ids') or [])
            j['sample_ids']=[s['id'] for s in data['samples'] if s.get('included',True) and (not selected or s['id'] in selected)]
        j['total']=len(j['sample_ids']); j['phase']='embeddings'
        return list(j['sample_ids'])
    ids=job_update(init)
    if not ids: raise ValueError('No included samples selected')
    cache=root/'embeddings'; cache.mkdir(exist_ok=True)
    model=None; processor=None; torch=None
    def load_model():
        nonlocal model,processor,torch
        if model is not None: return
        try:
            import torch as pytorch
            from transformers import AutoModel, AutoProcessor, CONFIG_MAPPING
            if 'embedding_gemma2' not in CONFIG_MAPPING: raise RuntimeError('Installed Transformers does not recognize embedding_gemma2')
            torch=pytorch
            processor=AutoProcessor.from_pretrained(EG2_MODEL,revision=EG2_REVISION)
            model=AutoModel.from_pretrained(EG2_MODEL,revision=EG2_REVISION,audio_config=None,dtype=torch.float32)
            model.eval()
        except Exception as exc:
            raise RuntimeError('Cannot load EmbeddingGemma 2. Verify compatible Transformers, model access and downloaded weights: '+str(exc)) from exc
    def encode(inputs):
        with torch.inference_mode():
            tokens=model(**inputs).last_hidden_state
            mask=inputs['attention_mask'].unsqueeze(-1).to(tokens.dtype)
            vector=(tokens*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1e-9)
            vector=torch.nn.functional.normalize(vector.float(),p=2,dim=-1)
        arr=vector.cpu().numpy()
        if arr.ndim!=2 or not np.isfinite(arr).all() or (np.linalg.norm(arr,axis=1)<.99).any(): raise ValueError('Model returned invalid embeddings')
        return arr
    def get_sample(sid):
        def get(data,j):
            if j.get('cancel_requested'): return None
            return next((dict(s) for s in data['samples'] if s['id']==sid),None)
        return job_update(get)
    vectors=[]; valid=[]
    for index,sid in enumerate(ids):
        s=get_sample(sid)
        if s is None: return
        path=image_path(root,s)
        if sha(path)!=s['sha256']: raise ValueError(f'Source changed: {path}')
        key=hashlib.sha256(json.dumps([EG2_MODEL,EG2_REVISION,EG2_PREPROCESS,s['sha256']]).encode()).hexdigest(); dest=cache/(key+'.npy')
        arr=None
        if dest.is_file():
            try:
                arr=np.load(dest,allow_pickle=False)
                if arr.ndim!=1 or not np.isfinite(arr).all() or not np.isclose(np.linalg.norm(arr),1,atol=.001): arr=None
            except (ValueError,OSError): arr=None
        if arr is None:
            load_model()
            with Image.open(path) as img:
                original=ImageOps.exif_transpose(img).convert('RGB')
                arr=encode(processor(images=[original],return_tensors='pt'))[0]
            with tempfile.NamedTemporaryFile(dir=cache,delete=False) as f:
                temp=f.name; np.save(f,arr,allow_pickle=False)
            os.replace(temp,dest)
        vectors.append(arr); valid.append(sid)
        def progress(data,j): j.update(cursor=index+1,processed=index+1,dimension=len(arr),model=EG2_MODEL,revision=EG2_REVISION)
        job_update(progress)
    matrix=np.stack(vectors)
    if mode=='cluster':
        from sklearn.cluster import KMeans
        k=int(opt.get('k',2))
        if k<1 or k>len(valid): raise ValueError('Cluster count must be between 1 and number of samples')
        labels=KMeans(n_clusters=k,random_state=int(opt.get('seed',42)),n_init=10).fit_predict(matrix)
        def apply(data,j):
            if j.get('cancel_requested'): return
            groups=[]
            for i in range(k):
                group={'id':uid(),'name':f'Grupo {i+1}','origin':'embeddinggemma2','job_id':j['id']}; data['groups'].append(group); groups.append(group)
            samples={s['id']:s for s in data['samples']}
            for sid,label in zip(valid,labels):
                s=samples.get(sid)
                if s is not None:
                    s['group_proposal']=groups[int(label)]['id']
            j['phase']='complete'
        job_update(apply)
    else:
        def get_classes(data,j): return [dict(c) for c in data['classes']]
        classes=job_update(get_classes)
        if not classes: raise ValueError('Create at least one class before classification')
        load_model()
        texts=['task: classification | query: '+c['name']+(': '+c['description'] if c.get('description') else '') for c in classes]
        text_vectors=encode(processor(text=texts,padding=True,return_tensors='pt'))
        if matrix.shape[1]!=text_vectors.shape[1]: raise ValueError('Cached image and text embedding dimensions differ')
        similarities=matrix@text_vectors.T
        def apply(data,j):
            if j.get('cancel_requested'): return
            samples={s['id']:s for s in data['samples']}; current={c['id'] for c in data['classes']}
            if not {c['id'] for c in classes}<=current or [{k:c.get(k,'') for k in ('id','name','description')} for c in data['classes']] != [{k:c.get(k,'') for k in ('id','name','description')} for c in classes]: raise ValueError('Classes changed during classification; retry')
            for sid,scores in zip(valid,similarities):
                s=samples.get(sid)
                if s is None: continue
                order=np.argsort(-scores); best=int(order[0]); gap=float(scores[best]-scores[int(order[1])]) if len(order)>1 else 2.0
                candidates=[{'class_id':classes[int(i)]['id'],'similarity':float(scores[int(i)])} for i in order]
                proposal={**candidates[0],'origin':'embeddinggemma2','model':EG2_MODEL,'revision':EG2_REVISION,'job_id':j['id'],'margin':gap,'alternatives':candidates[1:],'eligible':bool(float(scores[best])>=threshold and gap>=margin)}
                s['proposals']=[p for p in s.get('proposals',[]) if p.get('origin')!='embeddinggemma2']+[proposal]
            j['phase']='complete'
        job_update(apply)
