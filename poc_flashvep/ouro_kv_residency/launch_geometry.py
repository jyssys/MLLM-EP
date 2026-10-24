"""Read actual launch geometry. Do not substitute it for occupancy counters."""
import sqlite3
from .common import ROOT,write

def main():
    rows=[]
    for path in sorted((ROOT/'profiles').glob('profile_P*.sqlite')):
        db=sqlite3.connect(path);names=dict(db.execute('select id,value from StringIds'))
        sm,l2=db.execute('select smCount,l2CacheSize from TARGET_INFO_GPU where id=0').fetchone()
        for name,x,y,z,count in db.execute('select demangledName,gridX,gridY,gridZ,count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by demangledName,gridX,gridY,gridZ'):
            if 'flash_attn' in names[name]:
                rows.append(dict(run=path.stem,grid=[x,y,z],thread_blocks=x*y*z,calls=count,SMs=sm,L2_bytes=l2,
                    evidence='LIVE_MEASURED_OPERATOR',qualifier='Captured launch geometry. Not measured SM occupancy or memory bandwidth.'))
        db.close()
    write(ROOT/'launch_geometry.json',rows)

if __name__=='__main__':main()
