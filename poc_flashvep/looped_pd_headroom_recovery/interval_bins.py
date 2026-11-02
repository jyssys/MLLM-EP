"""Compute exact interval unions without rescanning every kernel per bin."""
import math


def bins(kernels,begin,end,width):
    intervals=sorted((max(begin,k['start_monotonic']),min(end,k['end_monotonic']))
        for k in kernels if k['end_monotonic']>begin and k['start_monotonic']<end)
    merged=[]
    for start,stop in intervals:
        if stop<=start:continue
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],stop)
        else:merged.append([start,stop])
    result=[];cursor=0
    for i in range(math.ceil((end-begin)/width)):
        start=begin+i*width;stop=min(start+width,end)
        while cursor<len(merged) and merged[cursor][1]<=start:cursor+=1
        j=cursor;occupied=0.
        while j<len(merged) and merged[j][0]<stop:
            occupied+=max(0,min(stop,merged[j][1])-max(start,merged[j][0]))
            j+=1
        result.append((start,stop,100*occupied/(stop-start)))
    return result
