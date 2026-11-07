from __future__ import annotations

import hashlib
import json


def semantic_graph(graph: dict) -> dict:
    """Return only executable graph state, independent of canvas presentation."""
    nodes=[]
    for item in graph.get("nodes",[]):
        data=item.get("data",{})
        nodes.append({"id":item.get("id"),"type":data.get("blockType"),"properties":data.get("properties",{})})
    node_types={item["id"]:item["type"] for item in nodes}
    target_orders={}
    edges=[]
    for item in graph.get("edges",[]):
        target=item.get("target");order=target_orders.get(target,0);target_orders[target]=order+1
        edge={"source":item.get("source"),"sourceHandle":item.get("sourceHandle"),"target":target,"targetHandle":item.get("targetHandle")}
        if node_types.get(target)=="concat":
            handle=str(item.get("targetHandle") or "")
            suffix=handle.rsplit("_",1)[-1]
            # New graphs carry an explicit, stable input port. Preserve the old
            # serialized order only for legacy connections without a port.
            edge["targetOrder"]=int(suffix) if handle.startswith("input_") and suffix.isdigit() else order
        edges.append(edge)
    return {"schemaVersion":3,"nodes":sorted(nodes,key=lambda value:str(value["id"])),"edges":sorted(edges,key=lambda value:(str(value["target"]),int(value["targetOrder"]) if "targetOrder" in value else str(value["targetHandle"]),str(value["source"]),str(value["sourceHandle"])))}


def semantic_hash(graph: dict) -> str:
    payload=json.dumps(semantic_graph(graph),sort_keys=True,separators=(",",":"),ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def comparison_hash(task_id:str,dataset:dict,metric:str) -> str:
    """Runs are comparable only when task, evaluation population and labels match."""
    contract={"task_id":task_id,"metric":metric,"dataset_id":dataset.get("id"),"dataset_revision":dataset.get("revision",1),"splits":dataset.get("splits"),"split_counts":dataset.get("splitCounts"),"classes":dataset.get("classes"),"output_shape":dataset.get("outputShape")}
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
