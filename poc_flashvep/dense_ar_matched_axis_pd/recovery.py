"""Resolve explicitly recorded external interruptions without rewriting evidence."""
import json
from .common import ROOT,digest

RECORD=ROOT/'state/daemon_recovery_20261118.json'


def record():
    return json.loads(RECORD.read_text()) if RECORD.exists() else {}


def resolve_name(name):
    return record().get('replacements',{}).get(name,name)


def acknowledged_failure(name):
    value=record()
    entry=value.get('interrupted_runs',{}).get(name)
    path=ROOT/'raw'/name/'dispatcher_result.json'
    return bool(value.get('validated_before_resume') and entry and path.exists()
        and value.get('replacements',{}).get(name)==entry['replacement_name']
        and digest(path)==entry['dispatcher_SHA256'])


def acknowledged_controller_stop():
    value=record();path=ROOT/'state/controller_stopped.json'
    return bool(value.get('validated_before_resume') and path.exists()
        and digest(path)==value.get('controller_stop_SHA256'))


def resolve_configuration(cfg):
    name=resolve_name(cfg['name'])
    if name==cfg['name']:return cfg
    assert acknowledged_failure(cfg['name']),'The interrupted receipt does not match its preserved recovery record'
    assert json.loads((ROOT/'configs'/(cfg['name']+'.json')).read_text())==cfg
    assert json.loads((ROOT/'raw'/cfg['name']/'configuration.json').read_text())==cfg
    replacement=dict(cfg,name=name)
    path=ROOT/'configs'/(name+'.json')
    if path.exists():assert json.loads(path.read_text())==replacement
    return replacement
