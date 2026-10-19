"""Compare only naturally reached token milestones. Never force EOS continuation."""
from poc_depth_cdb_factorial.contract import *
from poc_depth_cdb_factorial.scripts.analyze import records,paired_ratio

def main():
    cases={(d['case']['C'],d['case']['repeat'],d['case']['arm']):(d,{r['id']:r for r in values})
        for _,_,d,_,values in records() if d['case']['kind']=='speed' and not d.get('trace')}
    summaries=[]
    for C in [4,8,16,32,64]:
        for K in [32,64,128,256]:
            by_repeat=[]
            for repeat in range(5):
                if not all((C,repeat,a) in cases for a in ARMS):continue
                ids=cases[C,repeat,'A0'][0]['case']['ids']
                eligible=[i for i in ids if all(len(cases[C,repeat,a][1][i]['token_host_times_s'])>=K for a in ARMS)]
                if not eligible:continue
                times={a:max(cases[C,repeat,a][1][i]['token_host_times_s'][K-1] for i in eligible)
                    -cases[C,repeat,a][0]['first_submitted_s'] for a in ARMS}
                by_repeat.append(dict(repeat=repeat,N=len(eligible),ids=eligible,times_s=times))
            if not by_repeat:continue
            summaries.append(dict(C=C,K=K,eligible_by_repeat=[r['N'] for r in by_repeat],
                no_refill_hindsight=paired_ratio([r['times_s']['A1'] for r in by_repeat],[r['times_s']['A2'] for r in by_repeat]),
                refill_hindsight=paired_ratio([r['times_s']['A3'] for r in by_repeat],[r['times_s']['A4'] for r in by_repeat]),
                raw=by_repeat,
                scope='Time until all eligible naturally long requests reach K tokens. Full concurrent workload continues. Survivor-selection diagnostic, not cohort makespan.'))
    write(ROOT/'artifacts/equal_natural_progress.json',summaries)
    print('EQUAL_NATURAL_PROGRESS',len(summaries))

if __name__=='__main__':main()
