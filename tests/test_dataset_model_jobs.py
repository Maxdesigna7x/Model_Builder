"""Job safety tests with a simulated detector, not validation of model weights."""
import sys
from types import SimpleNamespace
from PIL import Image
from modelbuilder import dataset_project as ds
from modelbuilder.dataset_models import run, checkpoint
from modelbuilder.storage import create_project

class Scalar:
    def __init__(self, value): self.value=value
    def item(self): return self.value

class Coordinates:
    def __getitem__(self,index): return self
    def tolist(self): return [1.,1.,8.,8.]

def fixture_job(tmp_path,monkeypatch,cancel_during_prediction=False):
    monkeypatch.setenv('MODELBUILDER_WORKSPACE',str(tmp_path))
    project={'id':'model-job'};create_project(project);root=ds.root_of(project)
    path=tmp_path/'input.png';Image.new('RGB',(10,10),'red').save(path);item=ds.sample(path,'source')
    job={'id':'job','state':'running','operation':'yolo','options':{'action':'both'},'cursor':0,'errors':[]}
    ds.mutate(root,lambda d:d.update(samples=[item],jobs=[job]))
    def change(fn): return ds.mutate(root,lambda d:fn(d,d['jobs'][0]))
    class Detector:
        def __init__(self,name): self.names={0:'cat'}
        def predict(self,*args,**kwargs):
            if cancel_during_prediction: change(lambda d,j:j.update(cancel_requested=True))
            return [SimpleNamespace(names=self.names,boxes=[SimpleNamespace(cls=Scalar(0),conf=Scalar(.8),xyxy=Coordinates())])]
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Detector))
    return project,root,item,change

def test_yolo_cancel_during_inference_no_partial_result(tmp_path,monkeypatch):
    project,root,item,change=fixture_job(tmp_path,monkeypatch,True)
    run(root,ds.load(root)['jobs'][0],change)
    data=ds.load(root)
    assert data['samples'][0]['boxes']==[] and len(data['samples'])==1
    assert data['jobs'][0]['cursor']==0
    assert not list((root/'media'/'crops').glob('*.jpg'))

def test_yolo_resume_does_not_repeat_and_crop_proposal_preserves_human_label(tmp_path,monkeypatch):
    project,root,item,change=fixture_job(tmp_path,monkeypatch)
    run(root,ds.load(root)['jobs'][0],change)
    run(root,ds.load(root)['jobs'][0],change)
    data=ds.load(root); assert len(data['samples'])==2 and len(data['samples'][0]['boxes'])==1
    crop=data['samples'][1]; cat=data['classes'][0]['id']
    ds.update(project,{'classes':[{'name':'human'}]}); human=ds.load(root)['classes'][1]['id']
    ds.update(project,{'samples':[{'id':crop['id'],'class_ids':[human]}]})
    ds.update(project,{'accept_proposals':[crop['id']]})
    assert ds.load(root)['samples'][1]['class_ids']==[human] and cat!=human

def test_supported_checkpoint_names():
    assert checkpoint('YOLOv9','t')=='yolov9t.pt'
    assert checkpoint('YOLOv5u','n')=='yolov5nu.pt'
