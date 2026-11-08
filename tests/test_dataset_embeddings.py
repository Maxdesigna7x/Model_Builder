"""Cache and clustering behavior; these tests do not validate model weights."""
import hashlib, json
import numpy as np
from PIL import Image
from modelbuilder import dataset_project as ds
from modelbuilder.dataset_models import EG2_MODEL, EG2_REVISION, EG2_PREPROCESS, run_embeddings
from modelbuilder.storage import create_project

def test_cached_clustering_keeps_manual_group(tmp_path,monkeypatch):
    monkeypatch.setenv('MODELBUILDER_WORKSPACE',str(tmp_path))
    project={'id':'emb-test'}; create_project(project); root=ds.root_of(project)
    images=[]
    for i in range(3):
        path=tmp_path/f'{i}.png'; Image.new('RGB',(16,16),(i*80,0,0)).save(path)
        images.append(ds.sample(path,'source'))
    images[0]['group_id']='manual'
    def setup(d): d['samples']=images;d['jobs']=[{'id':'job','options':{'mode':'cluster','k':2},'operation':'embeddings','state':'running'}]
    ds.mutate(root,setup); (root/'embeddings').mkdir()
    for s,vector in zip(images,([1.,0.],[.99,.1],[0.,1.])):
        key=hashlib.sha256(json.dumps([EG2_MODEL,EG2_REVISION,EG2_PREPROCESS,s['sha256']]).encode()).hexdigest()
        vector=np.array(vector);np.save(root/'embeddings'/f'{key}.npy',vector/np.linalg.norm(vector))
    def job_update(fn):
        return ds.mutate(root,lambda d:fn(d,d['jobs'][0]))
    run_embeddings(root,ds.load(root)['jobs'][0],job_update)
    data=ds.load(root)
    assert len(data['groups'])==2 and data['samples'][0]['group_id']=='manual'
    assert all(s.get('group_proposal') for s in data['samples'])
    assert data['jobs'][0]['dimension']==2
    # A crash after applying groups but before final job status must not duplicate them.
    run_embeddings(root,data['jobs'][0],job_update)
    assert len(ds.load(root)['groups'])==2
