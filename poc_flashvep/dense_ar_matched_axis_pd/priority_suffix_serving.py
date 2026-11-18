"""Use a separate observer entry point without changing frozen inference."""
import subprocess


def main():
    previous=subprocess.Popen
    class Popen(previous):
        def __init__(self,args,*positional,**keywords):
            if isinstance(args,(list,tuple)) and '-m' in args:
                args=list(args);index=args.index('-m')+1
                if args[index] in ('poc_flashvep.dense_ar_slo_layer_pd.native',
                        'poc_flashvep.dense_ar_matched_axis_pd.worker'):
                    args[index]='poc_flashvep.dense_ar_matched_axis_pd.priority_suffix_worker'
            super().__init__(args,*positional,**keywords)
    subprocess.Popen=Popen
    from .serving import main as serving
    serving()


if __name__=='__main__':
    main()
