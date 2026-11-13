"""Create a separate executor copy from the immutable publication commit."""
import hashlib,json,subprocess
from pathlib import Path

REPO=Path('/home/esjung/MLLM-EP-github')
BASE='96414d7baac14dea2d697198b2c595850ae83661'
SOURCE=Path(__file__).parent
FILES=('__init__.py','common.py','native.py','dynamic_native.py','extension.py','dynamic_extension.py',
    'hidden_transport.py','output_priming.py','async_ingress.py','ack_retention.py','worker.py','credits.py',
    'token_protocol.py','policy.py','watch.py','boundaries.py','activity.py','run.py','analyze.py')


def main():
    result=[]
    for name in FILES:
        path=SOURCE/name;assert not path.exists(),path
        previous='poc_flashvep/dense_ar_dynamic_layer_pd/'+name
        data=subprocess.check_output(['git','show',BASE+':'+previous],cwd=REPO)
        text=data.decode().replace('poc_flashvep.dense_ar_dynamic_layer_pd','poc_flashvep.dense_ar_elastic_layer_pd')
        if name=='run.py':text=text.replace("('dense_ar_dynamic_layer_pd','dense_ar_layerwise_pd'", "('dense_ar_elastic_layer_pd','dense_ar_dynamic_layer_pd','dense_ar_layerwise_pd'")
        path.write_text(text)
        result.append(dict(path=previous,base_SHA256=hashlib.sha256(data).hexdigest(),
            snapshot_path=str(path.relative_to(REPO)),initial_snapshot_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
