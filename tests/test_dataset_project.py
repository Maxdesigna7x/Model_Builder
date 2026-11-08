import time
from pathlib import Path
import pytest
from PIL import Image
from modelbuilder.storage import create_project
from modelbuilder import dataset_project as ds

@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv('MODELBUILDER_WORKSPACE',str(tmp_path))
    project={'id':'dataset-test','projectType':'dataset'}
    create_project(project)
    return project

def wait(project,job):
    for _ in range(300):
        state=next(j for j in ds.read(project)['jobs'] if j['id']==job['id'])
        if state['state'] not in ('pending','running'): return state
        time.sleep(.02)
    pytest.fail('job timed out')

def test_import_export_and_duplicates(project,tmp_path):
    root=tmp_path/'images'; (root/'cats').mkdir(parents=True)
    Image.new('RGB',(80,60),'red').save(root/'cats'/'one.png')
    (root/'cats'/'two.png').write_bytes((root/'cats'/'one.png').read_bytes())
    (root/'cats'/'bad.png').write_bytes(b'bad')
    job=wait(project,ds.start(project,'images',{'path':str(root),'labels':True}))
    result=ds.read(project)
    assert job['state']=='completed' and len(job['errors'])==1
    assert result['total']==2 and len(result['summary']['duplicates'])==1
    assert len(result['classes'])==1
    exported=ds.export(project,{'format':'classification','seed':7})
    assert len({s['split'] for s in exported['samples']})==1
    assert len(list(Path(exported['path']).glob('*/*/*.jpg')))==2

def test_boxes_negative_and_pending(project,tmp_path):
    root=tmp_path/'images';root.mkdir()
    for i in range(3): Image.new('RGB',(80,60),(i*50,0,0)).save(root/f'{i}.png')
    assert wait(project,ds.start(project,'images',{'path':str(root)}))['state']=='completed'
    ds.update(project,{'classes':[{'name':'cat'}]}); result=ds.read(project); cid=result['classes'][0]['id']; samples=result['samples']
    ds.update(project,{'samples':[{'id':samples[0]['id'],'status':'annotated','boxes':[{'class_id':cid,'xywh':[.5,.5,.5,.5]}]},{'id':samples[1]['id'],'status':'negative'}]})
    exported=ds.export(project,{'format':'detection'})
    assert sum(exported['counts'].values())==2 and exported['omitted']['unreviewed_or_empty']==1
    labels=list(Path(exported['path']).glob('labels/*/*.txt'))
    assert len(labels)==2 and sorted(len(p.read_text().splitlines()) for p in labels)==[0,1]
    with pytest.raises(Exception): ds.update(project,{'samples':[{'id':samples[0]['id'],'boxes':[{'class_id':cid,'xywh':[0,0,1,1]}]}]})

def test_video_sampling_reuse(project,tmp_path):
    cv2=pytest.importorskip('cv2')
    import numpy as np
    video=tmp_path/'video.avi'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    if not writer.isOpened(): pytest.skip('MJPG codec unavailable')
    for i in range(30): writer.write(np.full((32,32,3),i*7,dtype=np.uint8))
    writer.release(); opts={'path':str(video),'mode':'frames','interval':5}
    assert ds.estimate(str(video),opts)['estimated']==6
    assert wait(project,ds.start(project,'video',opts))['state']=='completed'
    assert wait(project,ds.start(project,'video',opts))['state']=='completed'
    result=ds.read(project); assert result['total']==6
    assert [s['frame_index'] for s in result['samples']]==[0,5,10,15,20,25]

def test_job_recovery_and_run_token(project, monkeypatch):
    root=ds.root_of(project)
    def fake_launch(root,job): job.update(state='pending',pid=99999999,run_token='new',cancel_requested=False)
    monkeypatch.setattr(ds,'launch',fake_launch)
    job=ds.start(project,'images',{'path':'/nonexistent'})
    assert ds.read(project)['jobs'][0]['state']=='interrupted'
    assert ds.control(project,job['id'],'resume')['run_token']=='new'
    ds.worker(root,job['id'],'obsolete')
    assert ds.load(root)['jobs'][0]['state']=='pending'
    ds.control(project,job['id'],'cancel')
    ds.worker(root,job['id'],'new')
    assert ds.load(root)['jobs'][0]['state']=='cancelled'

def test_revisions_bulk_accept_and_alerts(project,tmp_path):
    root=tmp_path/'images';root.mkdir()
    Image.new('RGB',(24,24),'red').save(root/'a.png')
    (root/'b.png').write_bytes((root/'a.png').read_bytes())
    assert wait(project,ds.start(project,'images',{'path':str(root)}))['state']=='completed'
    ds.update(project,{'classes':[{'name':'cat'}]}); result=ds.read(project); cid=result['classes'][0]['id']; s=result['samples'][0]
    assert ds.read(project,filters={'alert':'duplicate'})['total']==2
    assert ds.read(project,filters={'alert':'small','min_size':20})['total']==0
    result=ds.update(project,{'samples':[{'id':s['id'],'_revision':s['_revision'],'proposals':[{'class_id':cid,'eligible':True,'origin':'embeddinggemma2'}]}]})
    with pytest.raises(Exception,match='DATASET_SAMPLE_CONFLICT'):
        ds.update(project,{'samples':[{'id':s['id'],'_revision':s['_revision'],'status':'negative'}]})
    assert result['revisions'][s['id']]>s['_revision']
    ds.update(project,{'accept_proposals':[s['id']]})
    assert ds.read(project)['samples'][0]['class_ids']==[cid]
    preview=ds.export(project,{'format':'classification'},preview=True)
    assert sum(preview['counts'].values())==1 and preview['omitted']['unassigned']==1
    assert not list((ds.root_of(project)/'exports').iterdir())

def test_group_proposals_never_override_confirmed(project):
    root=ds.root_of(project)
    def setup(d):
        d['groups']=[{'id':'human','name':'Human'},{'id':'auto','name':'Automatic'}]
        d['samples']=[{'id':'one','group_id':'human','group_proposal':'auto','proposals':[{'origin':'x'}]}, {'id':'two','group_proposal':'auto','proposals':[{'origin':'x'}]}]
    ds.mutate(root,setup)
    ds.update(project,{'accept_group_proposals':['one','two']})
    samples={s['id']:s for s in ds.load(root)['samples']}
    assert samples['one']['group_id']=='human' and samples['two']['group_id']=='auto'
    ds.update(project,{'discard_proposals':{'ids':['one'],'kind':'all'}})
    samples={s['id']:s for s in ds.load(root)['samples']}
    assert samples['one']['group_id']=='human' and not samples['one']['proposals'] and 'group_proposal' not in samples['one']

def test_responses_limit_export_samples_and_revision_ids(project):
    root=ds.root_of(project)
    def setup(data):
        data['samples']=[{'id':'a','status':'unreviewed','class_ids':[]}, {'id':'b','status':'unreviewed','class_ids':[]}]
        data['exports']=[{'id':'export','path':'somewhere','samples':[{'sample_id':'a'},{'sample_id':'b'}]}]
    ds.mutate(root,setup)
    result=ds.read(project)
    assert result['exports']==[{'id':'export','path':'somewhere','sample_count':2}]
    assert len(ds.load(root)['exports'][0]['samples'])==2
    revision=result['samples'][0]['_revision']
    untouched=ds.update(project,{'samples':[{'id':'a','_revision':revision,'status':'unreviewed'}]})
    assert untouched['revisions']=={'a':revision}
    changed=ds.update(project,{'samples':[{'id':'a','status':'negative'}]})
    assert changed['revisions']=={'a':revision+1}
    class_only=ds.update(project,{'classes':[{'name':'Cat'}]})
    assert class_only['revisions']=={}

def test_polling_omits_job_sample_ids_only(project):
    root=ds.root_of(project)
    def setup(data):
        data['jobs']=[{'id':'job','state':'completed','sample_ids':['a','b'],'classes':['cat'],'errors':[{'error':'bad'}],'processed':2,'total':2}]
    ds.mutate(root,setup)
    job=ds.read(project)['jobs'][0]
    assert 'sample_ids' not in job
    assert job['classes']==['cat'] and job['errors']==[{'error':'bad'}] and job['processed']==2 and job['total']==2
    assert ds.load(root)['jobs'][0]['sample_ids']==['a','b']

def test_durable_export_cancel_resume_snapshot_and_recovery(project,tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir()
    for i in range(3): Image.new('RGB',(50,40),(i*80,0,0)).save(image_dir/f'{i}.png')
    assert wait(project,ds.start(project,'images',{'path':str(image_dir)}))['state']=='completed'
    ds.update(project,{'classes':[{'name':'cat'}]}); data=ds.read(project); cid=data['classes'][0]['id']
    ds.update(project,{'samples':[{'id':s['id'],'class_ids':[cid],'status':'annotated'} for s in data['samples']]})
    root=ds.root_of(project); job={'id':'exporttest','operation':'export','state':'running','options':{'format':'classification','name':'out'}}
    ds.mutate(root,lambda d:d['jobs'].append(job))
    def change(fn):
        def apply(d):
            j=next(j for j in d['jobs'] if j['id']=='exporttest'); result=fn(d,j)
            if j.get('processed')==1 and not j.get('cancel_requested'): j['cancel_requested']=True
            return result
        return ds.mutate(root,apply)
    ds.export_worker(root,job,change)
    assert not (root/'exports'/'.pending-exporttest').exists()
    assert not (root/'exports'/'out-exporttest').exists()
    assert (root/'jobs'/'exporttest-export.json').exists()
    def prepare(d):
        j=next(j for j in d['jobs'] if j['id']=='exporttest');j['cancel_requested']=False
        d['samples'][0]['included']=False
    ds.mutate(root,prepare)
    def resumed(fn): return ds.mutate(root,lambda d:fn(d,next(j for j in d['jobs'] if j['id']=='exporttest')))
    ds.export_worker(root,job,resumed)
    data=ds.load(root); export=data['exports'][0]; j=data['jobs'][-1]
    assert len(export['samples'])==3 and j['processed']==j['total']==3
    assert len(list((root/'exports'/'out-exporttest').glob('*/*/*.jpg')))==3
    # Simulate a crash after directory publication but before manifest registration.
    ds.mutate(root,lambda d:d.update(exports=[]))
    ds.export_worker(root,job,resumed)
    assert len(ds.load(root)['exports'])==1

def test_export_job_runs_asynchronously(project,tmp_path):
    folder=tmp_path/'images';folder.mkdir();Image.new('RGB',(40,40),'red').save(folder/'one.png')
    assert wait(project,ds.start(project,'images',{'path':str(folder)}))['state']=='completed'
    ds.update(project,{'classes':[{'name':'cat'}]});state=ds.read(project)
    ds.update(project,{'samples':[{'id':state['samples'][0]['id'],'class_ids':[state['classes'][0]['id']]}]})
    job=wait(project,ds.start(project,'export',{'format':'classification'}))
    assert job['state']=='completed' and job['processed']==job['total']==1
    assert Path(job['result']['path'],'manifest.json').is_file()
    assert 'samples' not in job['result']
