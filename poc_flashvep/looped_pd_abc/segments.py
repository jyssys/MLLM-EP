"""Merge adjacent, already-valid source/destination ranges. This is generic packing."""
def coalesce(source,target):
 if len(source)!=len(target):raise ValueError('Source and destination descriptor counts differ')
 pairs=sorted(zip(source,target),key=lambda pair:(pair[0][2],pair[0][0]))
 result=[]
 for src,dst in pairs:
  if src[1]!=dst[1]:raise ValueError('Descriptor sizes differ')
  if result:
   a,b=result[-1]
   if a[2]==src[2] and b[2]==dst[2] and a[0]+a[1]==src[0] and b[0]+b[1]==dst[0]:
    result[-1]=((a[0],a[1]+src[1],a[2]),(b[0],b[1]+dst[1],b[2]));continue
  result.append((src,dst))
 return [a for a,b in result],[b for a,b in result]
