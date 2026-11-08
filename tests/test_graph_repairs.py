from copy import deepcopy
import json
import subprocess
from pathlib import Path

import pytest
import torch

from modelbuilder.engine import handle
from modelbuilder.errors import BackendError
from modelbuilder.models import GraphModel


def node(id, kind, **properties):
    return {"id":id,"type":"modelNode","position":{"x":0,"y":0},"data":{"blockType":kind,"label":id,"properties":properties}}


def chain(*nodes):
    return {"nodes":list(nodes),"edges":[{"id":f"{a['id']}-{b['id']}","source":a['id'],"target":b['id']} for a,b in zip(nodes,nodes[1:])]}


def request(graph, architecture='mlp', task='tabular.classification', input_shape=None, output_shape=None):
    return {"action":"graph.solutions","graph":graph,"architecture":architecture,"task_id":task,"dataset":{"inputShape":input_shape or [4],"outputShape":output_shape or [3]}}


def assert_repaired(req):
    original=deepcopy(req)
    rng=torch.random.get_rng_state().clone()
    reply=handle(req)
    assert req==original, 'proposals never mutate the submitted graph or dataset'
    assert torch.equal(rng,torch.random.get_rng_state()), 'repair probing does not perturb RNG state'
    assert reply['status']=='repairable', reply
    solution=reply['solutions'][0]
    assert solution['steps']
    validation=handle({**req,'action':'graph.validate','graph':solution['graph']})
    assert validation['valid'] and validation['backward']
    return reply, solution


def test_invalid_dropout_and_output_head_are_repaired_together():
    graph=chain(node('input','input'),node('drop','dropout',p=2),node('head','linear',out_features=7),node('output','output'))
    reply, solution=assert_repaired(request(graph))
    assert reply['problem']['node_id']=='drop'
    assert {s.get('property') for s in solution['steps']} >= {'p','out_features'}


def test_broken_reshape_and_output_rank():
    graph=chain(node('input','input'),node('head','linear',out_features=12),node('reshape','reshape',shape='5,5'),node('output','output'))
    assert_repaired(request(graph))


@pytest.mark.parametrize('architecture,task,shape',[
 ('mlp','tabular.classification',[4]),('cnn','image.classification',[3,8,8]),
 ('cnn1d','sequence.classification',[8,3]),('lstm','sequence.classification',[8,3]),
 ('transformer','sequence.classification',[8,3]),('transformer_causal','text.language_model',[8]),
 ('vit','image.classification',[3,8,8]),('unet','image.segmentation.binary',[3,8,8]),
 ('autoencoder','tabular.reconstruction',[4]),
])
def test_all_nine_architectures_offer_verified_repairs(architecture,task,shape):
    script="""
      import {templateFor,syncOutputContract} from './frontend/src/catalog.ts';
      const [arch,task,inputShape,outputShape]=JSON.parse(process.argv[1]);
      const g=templateFor(arch,task);g.nodes=syncOutputContract(g.nodes,g.edges,{inputShape,outputShape,options:{vocab_size:64,max_length:8}});
      console.log(JSON.stringify(g));
    """
    output=[8,64] if task=='text.language_model' else [1,8,8] if architecture=='unet' else [4] if architecture=='autoencoder' else [3]
    result=subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',script,json.dumps([architecture,task,shape,output])],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
    graph=json.loads(result.stdout)
    # A deliberately invalid dropout is an architecture-independent execution failure.
    terminal=next(n for n in graph['nodes'] if n['data']['blockType']=='output')
    edge=next(e for e in graph['edges'] if e['target']==terminal['id'])
    graph['nodes'].append(node('broken-dropout','dropout',p=1.5))
    edge['target']='broken-dropout'
    graph['edges'].append({'id':'repair-test-output','source':'broken-dropout','target':terminal['id']})
    assert_repaired(request(graph,architecture,task,shape,output))


def test_transformer_heads_uses_actual_input_dimension():
    graph=chain(node('input','input'),node('projection','sequence_projection',d_model=10),node('attention','transformer_encoder',heads=4,num_layers=1,dim_feedforward=16),node('pool','sequence_pool'),node('head','linear',out_features=3),node('output','output'))
    reply,solution=assert_repaired(request(graph,'transformer','sequence.classification',[4,3]))
    assert reply['problem']['node_id']=='attention'
    assert next(n for n in solution['graph']['nodes'] if n['id']=='attention')['data']['properties']['heads']==2


@pytest.mark.parametrize('kind',['add','concat'])
def test_unequal_branch_shapes_are_adapted_on_the_offending_edge(kind):
    graph={'nodes':[node('input','input'),node('a','conv2d',out_channels=4,kernel_size=1),node('b','conv2d',out_channels=8,kernel_size=1,stride=2),node('merge',kind,axis=1),node('pool','adaptiveavgpool2d',output_size=1),node('flat','flatten'),node('head','linear',out_features=3),node('output','output')], 'edges':[]}
    for a,b,port in [('input','a',None),('input','b',None),('a','merge','input_0'),('b','merge','input_1'),('merge','pool',None),('pool','flat',None),('flat','head',None),('head','output',None)]:graph['edges'].append({'id':f'{a}-{b}','source':a,'target':b,'targetHandle':port})
    _,solution=assert_repaired(request(graph,'cnn','image.classification',[3,8,8]))
    assert any(n['data']['blockType']=='resize2d' for n in solution['graph']['nodes'])
    assert next(e for e in solution['graph']['edges'] if e['target']=='merge' and e.get('targetHandle')=='input_1')['source'].startswith('repair-')


def test_invalid_kernel_and_groups():
    graph=chain(node('input','input'),node('conv','conv2d',out_channels=5,kernel_size=9,padding=0,groups=2),node('pool','adaptiveavgpool2d',output_size=1),node('flat','flatten'),node('head','linear',out_features=3),node('output','output'))
    assert_repaired(request(graph,'cnn','image.classification',[3,4,4]))


def test_unknown_error_never_offers_unverified_graph():
    graph=chain(node('input','input'),node('unknown','unsupported'),node('output','output'))
    reply=handle(request(graph))
    assert reply['status']=='unsupported' and reply['solutions']==[]


def test_valid_graph_does_not_offer_changes():
    graph=chain(node('input','input'),node('head','linear',out_features=3),node('output','output'))
    assert handle(request(graph))['status']=='valid'


def test_operation_constructor_errors_have_stable_node_context():
    graph=chain(node('input','input'),node('bad','dropout',p=5),node('head','linear',out_features=3),node('output','output'))
    with pytest.raises(BackendError) as exc:handle({**request(graph),'action':'graph.validate'})
    assert exc.value.context['node_id']=='bad'
    assert exc.value.context['input_shapes']==[[4]]


def test_positional_length_failure_preserves_its_error_code():
    graph=chain(node('input','input'),node('projection','sequence_projection',d_model=8),node('positions','positional_encoding',max_length=2),node('pool','sequence_pool'),node('head','linear',out_features=3),node('output','output'))
    reply,_=assert_repaired(request(graph,'transformer','sequence.classification',[6,3]))
    assert reply['problem']['code']=='MODEL_SEQUENCE_TOO_LONG'


def test_minimal_conv_fix_does_not_add_defaults_or_change_channel_count():
    graph=chain(node('input','input'),node('conv','conv1d',kernel_size=2,stride=0),node('flat','flatten'),node('head','linear',out_features=3),node('output','output'))
    _,solution=assert_repaired(request(graph,'cnn1d','sequence.classification',[4,8]))
    conv=next(n for n in solution['graph']['nodes'] if n['id']=='conv')
    assert conv['data']['properties']=={'kernel_size':2,'stride':1}


def test_structured_token_input_does_not_mask_original_error():
    graph=chain(node('input','input'),node('tokens','hf_token_processor'),node('bad','hf_unknown'),node('output','output'))
    with pytest.raises(BackendError) as exc:GraphModel(graph,[4],torch.long)
    assert exc.value.code=='MODEL_BLOCK_NOT_SUPPORTED'
    assert exc.value.context['node_id']=='bad'
    assert exc.value.context['input_shapes']==[[4]]
