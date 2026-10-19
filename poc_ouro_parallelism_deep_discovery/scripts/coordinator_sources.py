"""Save primary metadata and bounded coordinator source/search notes."""
import datetime
from html.parser import HTMLParser
import urllib.request
from poc_ouro_parallelism_deep_discovery.contract import *

class Metadata(HTMLParser):
    def __init__(self):super().__init__();self.values={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs);name=attrs.get('name','')
        if tag=='meta' and name.startswith('citation_'):
            self.values.setdefault(name,[]).append(attrs.get('content',''))

def main():
    target=ROOT/'artifacts/coordinator_sources.json'
    assert not target.exists()
    specifications=[
        ('M01','2510.25741v6',['Section3.1–3.2','Table2','Section5.4.2','Section7.3'],
         'Latest metadata and PDF selected sections read. HTMLv6 returned404. Earlier v4 HTML was read first.',
         'Dense shared core, original gate, distinct prefill KV, and native draft/verify discussion. Cache variants are not interchangeable references.'),
        ('M02','2604.26294v1',['SectionsII–IV','SectionV','SectionsVI–VII'],
         'Method, communication and memory analysis, evaluation setting, and limitations read.',
         'Generic TP/SP folding moves weight shards and KV. MI300X long-context results are not H100 R4 evidence.'),
        ('M03','2211.05102v1',['Introduction','Parallel partitioning overview'],
         'Partial read. Detailed implementation and complete appendix remain unaudited.',
         'Generic inference partitioning and analytical efficiency model. TPU/PaLM/int8 figures are not BF16 Ouro measurements.'),
        ('M04','2602.09721v1',['AFD architecture','Communication roofline','Hardware-model scope'],
         'Selected architecture, cost boundaries, and limitations read.',
         'Attention/FFN disaggregation must charge both activation directions, batch readiness, and synchronization. The target models are MoE, not dense Ouro.'),
        ('M05','2601.21351v1',['Four-phase execution model','Analytical assumptions','Evaluation scope'],
         'Selected cost model and simulation assumptions read.',
         'Attention, communication, FFN, and return communication form distinct serial dependencies. Analytical provisioning is not our measured architecture.'),
        ('M06','2102.07988v1',['Section3.1–3.3','Equations4–6','Algorithm1'],
         'Pinned v1 method and cost analysis read. Latest body is not claimed.',
         'Causal token slicing and dynamic partition costs predate the proposed recurrence/chunk schedule. Training results do not prove Ouro inference gains.'),
        ('M07','2606.18023v1',['Section2.2','Section3.1 Training Infrastructure'],
         'Selected mechanism and training implementation description read. Repository README inspected, no executable runtime audit.',
         'Tied logical unrolling and physical-layer colocation already appear in a trained PLT system. Its changed offsets and KV semantics are not a drop-in pretrained Ouro transform.'),
    ]
    sources=[]
    for key,identifier,locators,audit,claim in specifications:
        before=datetime.datetime.now(datetime.timezone.utc).isoformat()
        request=urllib.request.Request('https://arxiv.org/abs/'+identifier,headers={'User-Agent':'Ouro research metadata audit'})
        with urllib.request.urlopen(request,timeout=30) as response:body=response.read(1024*1024)
        parser=Metadata();parser.feed(body.decode())
        assert parser.values.get('citation_title'),identifier
        sources.append(dict(source_id=key,identifier='arXiv:'+identifier,url='https://arxiv.org/abs/'+identifier,
            title=parser.values['citation_title'][0],authors=parser.values.get('citation_author',[]),
            primary_metadata=parser.values,metadata_access_start_utc=before,
            metadata_access_end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            content_access_window_utc=['2026-10-19T09:52:53Z','2026-10-19T10:49:00Z'],
            timestamp_note='Individual earlier reading timestamps were not captured. The interval bounds the observed coordinator sequence.',
            locators=locators,audit_depth=audit,claim=claim,evidence_class='SOURCE_REPORTED',runtime_execution='NOT_RUN'))
    sources.append(dict(source_id='M08',title='DeepSpeed Pipeline Parallelism tutorial',
        authors=['DeepSpeed project'],url='https://www.deepspeed.ai/tutorials/pipeline/',
        version='Live documentation read2026-10-19, page reports update2026-10-19',
        locators=['Tied Layers','DualPipeV Schedule'],
        claim='TiedLayerSpec names repeated layers. Owning stages replicate tied weights. This is a generic alias/placement comparison, not an Ouro inference implementation.',
        audit_depth='Selected documentation read. Source revision not pinned.',evidence_class='SOURCE_REPORTED',runtime_execution='NOT_RUN'))
    searches=[
        'Ouro looped transformer shared weights distributed inference tensor parallel recurrent model',
        'weight stationary recurrent transformer multi GPU inference shared weights pipeline',
        'communication avoiding tensor parallel inference layer loop normalization distributed residual',
        'ByteDance Ouro-2.6B model inference vllm recurrent adaptive gate',
        '"reentrant" "pipeline" "inference" transformer shared weight',
        '"shared weights" "inference" "multi-GPU" scheduling',
        'attention feed forward disaggregation inference AFD models weights KV',
        '"Folding Tensor and Sequence Parallelism"',
        'site:arxiv.org reentrant pipeline language inference',
        'site:arxiv.org attention FFN disaggregation inference',
        'site:arxiv.org Folding Tensor Sequence Parallelism',
        'site:arxiv.org shared weights pipeline inference',
        '"looped transformer" "cyclic" pipeline inference',
        '"reentrant" pipeline "inference" shared weights transformer',
        '"weight sharing" "interleaved pipeline" inference',
        '"tied" "pipeline parallelism" inference repeated layers',
        '"LoopCoder-v2" "Only Loop Once" github',
        '"TeraPipe" "token-level" "causal" pipeline',
        '"cyclic pipeline parallelism" inference language',
        '"reentrant" pipeline transformer serving weights',
    ]
    write(target,dict(sources=sources,searches=[dict(search_id=f'MQ{i+1:02d}',query=q,engine='web.run',
        execution_window_utc=['2026-10-19T09:52:53Z','2026-10-19T10:49:00Z'],
        timestamp_note='Query-specific timestamp not captured. This is an observed sequence window, not an invented exact time.',
        negative_result_rule='Not located in these queries does not prove absence of prior work.') for i,q in enumerate(searches)],
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('COORDINATOR_SOURCE_NOTES',len(sources),len(searches))

if __name__=='__main__':main()
