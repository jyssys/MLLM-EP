"""Freeze distinct mechanisms before candidate headroom or quality selection."""
import datetime
from poc_ouro_parallelism_deep_discovery.contract import *

CARDS=[
 dict(id='C01',title='Ready physical-operator workers',family='A',axis='recurrence hybrid',
  mechanism='Group already-ready tasks by immutable physical weight identity instead of logical recurrent depth.',
  sequence='Request-local complete stack -> published ready tasks -> tied operator workers -> named state owners.',
  concurrent='Different requests or causally ready prompt chunks. Same-token recurrent states are not independent.',
  saving='Possible shared GEMM weight access and reduced physical weight replicas.',
  added='Activation routing, queueing, batching delay, cache-owner reads, and fill/drain.',
  exactness='Numerically reordered exact graph',prior='CDB, grouped GEMM, aliased unrolled PP, LoopSpec mixed-depth batching',
  falsify='Enumerate the legal frontier. Reject a claimed C1 benefit if it requires unavailable predecessor states.',
  next='Use actual captured inputs for an ideal-ready batch-versus-replica operator test before any worker engine.'),
 dict(id='C02',title='KV-stationary remote MLP pool',family='B',axis='recurrence hybrid',
  mechanism='Keep attention state on its owner and send ready normalized activations to shared MLP workers.',
  sequence='Local attention and MLP -> local attention -> remote MLP round trip -> exact residual and norm.',
  concurrent='Independent state owners can supply different ready requests or depths to one physical MLP batch.',
  saving='Possible larger MLP batches without moving historical KV.',
  added='Both activation messages, packing, queueing, return dependency, and attention-worker idle time.',
  exactness='Numerically reordered exact graph',prior='Generic attention–FFN disaggregation and strong tied CDB/TP controls',
  falsify='Measure local MLP, remote round trip, and TP MLP on identical pretrained weights and actual inputs.',
  next='Prune any layout whose round trip loses before state/layout overhead or whose gain is only generic AFD.'),
 dict(id='C03',title='Versioned depth-address arena',family='B/C',axis='recurrence hybrid',
  mechanism='Use row-specific depth addresses so ready tokens at different recurrent phases can share an operator launch.',
  sequence='Scalar-slot launch and terminal copies -> physical-layer arena -> immutable per-row page table -> private append.',
  concurrent='Already-ready rows with distinct depth histories, not dependent depths of one current token.',
  saving='Possible mixed-depth batching and adaptive terminal-copy removal.',
  added='Address tables, metadata publication, token-page fragmentation, graph inputs, and allocator lifetime management.',
  exactness='Intended exact state representation',prior='WaveFront per-row depth addressing, paged attention, immutable page aliases',
  falsify='Prove logical address identity for mixed exit histories. Test supported page sizes and account for table overhead.',
  next='Separate mixed-depth addressing from no-copy aliases. Fixed R4 cannot alias unequal computed states.'),
 dict(id='C04',title='Loop-closed distributed layout',family='C',axis='recurrence hybrid',
  mechanism='Carry a useful sharded residual layout across operator and loop boundaries without repeating avoidable redistribution.',
  sequence='All-reduced residual per sublayer -> sharded residual with exact norm statistics -> consumer-compatible layout.',
  concurrent='Feature or operator tiles of an already-ready state. No guessed next-depth input.',
  saving='Only a collective whose real consumer can operate on the retained layout.',
  added='Norm statistics, deferred gathers or reductions, layout conversion, and possibly changed GEMM shapes.',
  exactness='Numerically reordered exact algebra until native BF16 casts are checked',prior='Generic TP/SP layout propagation, collective fusion, and TSP',
  falsify='Use two-layer/two-loop algebra checks. Name where every deferred collective reappears.',
  next='Do not carry unreduced sums through nonlinear functions. Compare a valid rewrite with generic fusion.'),
 dict(id='C05',title='Finite-trajectory parallel correction',family='E',axis='potential new execution axis',
  mechanism='Evaluate versioned recurrent trajectory guesses concurrently and charge all correction and KV repair work.',
  sequence='Four exact sequential passes -> causal initialization -> parallel residual/derivative rounds -> validation -> KV commit.',
  concurrent='Map evaluations at different depths use guessed states. They are solver tasks, not exact ready successors.',
  saving='A smaller charged number of correction rounds could shorten recurrence span.',
  added='Jacobian or JVP work, reductions, solver vectors, fresh scratch KV, validation, and full internal KV repair.',
  exactness='Approximate until a sufficient exact or quality-controlled correction/commit criterion passes',
  prior='DEER, ELK/quasi-DEER, ParaDiGMS, SNLP, and recurrent-depth parallel samplers',
  falsify='Check the finite-R4 break-even inequality against competent same-P TP, not only sequential one-GPU work.',
  next='Reject variants that require four expensive rounds or merely reproduce a known numerical layer-parallel method.'),
 dict(id='C06',title='Structured base-plus-fresh-residual projection',family='F',axis='potential component execution axis',
  mechanism='Separate a ready linear contribution from a later fresh residual while preserving all nonlinear dependencies.',
  sequence='One projection after a complete input -> base projection beside producer -> fresh residual projection -> exact merge.',
  concurrent='A contribution with a truly ready input can run beside its independent producer.',
  saving='Only exposed projection span, not the cost of both complete dense matrix products.',
  added='A second dense GEMM, normalization statistics, residual transfer, merges, and BF16 reordering.',
  exactness='Real-arithmetic identity, numerically reordered in BF16',prior='Exact linear distributivity and nearby FlashLoop approximation controls',
  falsify='Charge W*delta as a full GEMM unless a measured structure supports less work.',
  next='Run an algebra control and a pretrained operator cost test. Do not reuse a stale whole hidden state.'),
 dict(id='C07',title='Verified shared-depth branch ownership',family='D',axis='potential verified execution axis',
  mechanism='Distribute only useful verified token branches while retaining tied weight and versioned cache economics.',
  sequence='Serial target token -> shared-prefix draft/verify lanes -> remote ownership -> accept/prune -> authoritative commit.',
  concurrent='Future-token branches are speculative until target verification. Recurrent depths of one token remain chained.',
  saving='Accepted useful work started before the original token boundary.',
  added='Rejected branches, duplicate weight reads, remote prefix state, winner messages, rollback, and commit.',
  exactness='Validated speculative only after an exact greedy or distribution-preserving protocol passes',
  prior='LoopSpec, WaveFront, generic speculative decoding, and LoPA winner-cache synchronization',
  falsify='Compare co-batched branches with full-replica split and ordinary TP of the same branch work.',
  next='A single-GPU prior lacking multi-GPU code is not novelty. Require a new ownership transformation with measured benefit.'),
 dict(id='C08',title='Lossless recurrent-difference transport',family='F/G',axis='potential representation execution axis',
  mechanism='Transmit exact bit differences between versions only when encoded payload and processing beat full transfer.',
  sequence='Fresh full activation transfer -> compare named versions -> lossless encode -> transfer -> exact reconstruct.',
  concurrent='Encoding tiles and independent messages can overlap only after their actual input versions exist.',
  saving='Potential activation traffic, not an assumed reduction in compute or mathematical precision.',
  added='Previous buffers, XOR/encoding scans, metadata, variable payload coordination, decoding, and synchronization.',
  exactness='Lossless only with bitwise reconstruction. BF16 numerical subtraction alone is not lossless.',
  prior='Generic lossless communication compression and cross-loop residual methods',
  falsify='Measure exact equality, bit entropy, encoded size, and both processing directions on actual captured states.',
  next='Reject if payload reduction cannot beat small-message startup or if the same effect exists without recurrence.'),
 dict(id='C09',title='Certified partial-head decisions',family='G',axis='potential decision execution axis',outside=True,
  mechanism='Publish an exact greedy token after a valid bound proves that unfinished head tiles cannot change its winner.',
  sequence='Full LM head then argmax -> partial head tiles -> interval proof -> token publication.',
  concurrent='Independent vocabulary or feature tiles of the final actual normalized hidden state.',
  saving='Only the unfinished head tail on the next-token critical path.',
  added='Bound metadata, remaining-norm calculation, certificate reduction, and BF16 error envelopes.',
  exactness='Exact greedy only with valid finite-precision bounds. Sampling needs a separate guarantee.',
  prior='Generic branch-and-bound dot products and vocabulary-parallel heads',
  falsify='Test certificates on real head inputs. Charge all weight/norm reads and compare fused full-head argmax.',
  next='Reject a recurrence-specific claim unless the repeated model provides a distinct advantage beyond generic heads.'),
 dict(id='C10',title='Historical KV rematerialization frontier',family='G',axis='potential state execution axis',outside=True,
  mechanism='Store each actual depth/layer projection input and regenerate historical KV on auxiliary workers.',
  sequence='Stored full KV -> immutable residual checkpoints -> remote projections and RoPE -> versioned attention chunks.',
  concurrent='Historical checkpoint projections have known inputs and can precede the next query.',
  saving='Capacity or exposed remote KV transfer, not automatically local in-memory attention time.',
  added='History-sized K/V projections, scratch KV, weight residency, merge, and publication.',
  exactness='Intended exact algebra with GEMM-shape and KV parity still required',prior='Residual Stream and CacheFlow restoration',
  falsify='Use actual H2048/L24-or48 shapes to compare full-history reprojection with avoided state reads or transfer.',
  next='Reject if arithmetic and communication dominate or a generic residual-cache design explains all benefit.'),
 dict(id='C11',title='Quorum linear projection workers',family='G',axis='potential redundancy execution axis',outside=True,
  mechanism='Use redundant linear transforms to finish before a real slow rank while preserving required outputs.',
  sequence='Wait for all TP pieces -> encoded redundant pieces -> quorum -> complete linear decode -> nonlinear consumer.',
  concurrent='Redundant transforms consume the same ready input. No recurrent predecessor is removed.',
  saving='Only a measured rank tail larger than added redundant work and reconstruction.',
  added='Encoded weights, extra GEMMs, fanout, decoding, numerical control, and memory.',
  exactness='Real-arithmetic coding identity, native BF16 equivalence unresolved',prior='Short-Dot and generic coded computation',
  falsify='Look for a genuine baseline tail first. Do not create artificial stragglers or count the slowest rank twice.',
  next='Reject if dedicated H100 tails are too small or the BF16 encoded function changes.'),
 dict(id='C12',title='Committed attention-feature summaries',family='F/G',axis='potential feature execution axis',
  mechanism='Maintain fixed non-learned historical attention features and distribute current-query contractions.',
  sequence='Read complete historical KV -> append fixed feature summaries after commit -> query contraction and reduction.',
  concurrent='Feature updates use committed known K/V while future query computation proceeds elsewhere.',
  saving='Possible history reads under a separately established long-context bottleneck.',
  added='Feature expansion, summary state, normalization, current-token contribution, query/output communication, and approximation.',
  exactness='Approximate softmax unless an exact sufficient representation is proved',prior='Performer features, generic linear attention, and CP',
  falsify='Measure actual attention error and required state size before any free-running quality pilot.',
  next='Reject if quality collapses, exactness requires full KV, or generic kernel approximation explains all value.'),
 dict(id='C13',title='Exact-input continuation beside gate',family='D/A',axis='potential conditional execution axis',
  mechanism='Evaluate the unchanged exit gate beside provisional next-pass work from the already completed current state.',
  sequence='Complete pass -> gate -> successor pass -> complete pass -> gate and scratch successor -> commit or discard.',
  concurrent='Gate and continuation share a ready exact input. The successor still waits for the current pass.',
  saving='Only exposed gate or host-control delay for continuing tokens.',
  added='Wasted continuation, remote activation, replica residency, cancellation, scratch KV, and commit.',
  exactness='Verified conditional execution with original gate and private scratch',prior='Generic conditional graphs and speculative control execution',
  falsify='Measure an on-device gate/control baseline before comparing against an avoidable host round trip.',
  next='Mark MODE_SPECIFIC because fixed R4 has no adaptive gate wait to eliminate.'),
 dict(id='C14',title='Tied causal depth–chunk frontier control',family='A',axis='generic-control hypothesis',
  mechanism='Pipeline causally ready prefill chunks through complete recurrent passes or stationary physical layers.',
  sequence='Full prompt passes -> legal chunk/depth DAG with matching loop norms and depth KV.',
  concurrent='Different ready prompt chunks after their required layer/depth KV predecessors finish.',
  saving='Conventional prefill pipeline overlap and possible tied-weight capacity effects.',
  added='Chunk GEMM inefficiency, fill/drain, activation transfers, wrap-around, and decode transition.',
  exactness='Numerically reordered exact causal graph',prior='TeraPipe, generic chunked PP, and the existing failed wavefront novelty test',
  falsify='Compare edge-preserving aliased logical unrolling under identical placement and memory.',
  next='Do not rerun the old generic graph as a new method. Reopen only for a measured, distinct tied-memory optimum.'),
]

def main():
    target=ROOT/'artifacts/candidate_register_initial.json'
    assert not target.exists(),'Preserve the frozen initial candidate register.'
    for card in CARDS:
        card['revision']=1
        card['initial_status']='UNTESTED_HYPOTHESIS'
        card['both_size_mode_scope']={'1.4B_fixed':'scope target','1.4B_adaptive':'scope target',
            '2.6B_fixed':'scope target','2.6B_adaptive':'tiny runtime probe passed, quality unvalidated',
            'P2':'physical probe pending','P4':'physical probe pending','P8':'projection only'}
    write(target,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cards=CARDS,
        criterion='Distinct mechanisms, not a guarantee of novelty. No candidate headroom outcome selects this list.',
        distinct_family_count=7,outside_seed_ids=['C09','C10','C11'],
        candidate_count_is_not_time_accounting=True))
    for c in CARDS:
        text=f'''# {c['id']}: {c['title']}

## 1. Identity

Revision 1. Search family: {c['family']}. Initial category: {c['axis']}.

## 2. Mechanism

{c['mechanism']}

## 3. Relevant model property

The audited Base checkpoints share physical weights across four passes but retain depth-indexed state.
See `reports/ouro_semantics_audit.md` for pinned source locations.

## 4. Target scope

Both Base sizes, fixed R4, and the unchanged adaptive gate are scope targets, not demonstrated gains.
Primary C is 1/2/4/8. Contexts are 1K/4K/8K. P2/P4 require physical tests. P8 remains projected.

## 5. Baseline and saved cost

Compare mode-matched native 1GPU and the strongest correct same-P conventional baseline.
The hypothesized saving is: {c['saving']}

## 6. Operation sequence

{c['sequence']}

## 7. Legal DAG and ownership

Each task must identify request, token/chunk, recurrent depth, physical layer, layout, and version.
Weights are immutable. Historical KV belongs to its matching committed request/depth/layer prefix.
Current KV remains private until its complete input and commit rule pass.
The generic validator is `scripts/verify_dag.py`. This card is not a completed candidate-specific executable DAG.

## 8. Source of concurrency

{c['concurrent']}

## 9. Exactness and quality

{c['exactness']}.
Quality remains NOT_RUN for a new mechanism.
Use the frozen free-running calibration and held-out protocol before a quality-preservation claim.

## 10. State and memory

Charge unique and replicated weights, full required depth KV, current activations, graph buffers, and candidate scratch separately.
For H=2048, full BF16 KV is 786432 bytes/token at L24/R4 and 1572864 at L48/R4.
These are structural capacities, not measured HBM traffic.

## 11. Messages and topology

Only physical 4/5/6/7 are eligible for P2/P4.
Every message needs a producer, consumer, byte count, and completion dependency.
No zero-cost KV movement or unsupported eight-GPU result is permitted.

## 12. Added costs and critical path

{c['added']}
Use a resource-constrained path. Do not add overlapping intervals or replace serial work with a maximum without ready inputs.

## 13. Nearest work

{c['prior']}.
The source audits determine overlap. The initial category is not a novelty claim.

## 14. Relation to past failures

The scope ledger preserves previous stale/delta quality failure, generic attention/wavefront findings, and scheduling uncertainty.
No prior negative result disappears through renaming.

## 15. Smallest falsification

{c['falsify']}

## 16. Results and uncertainty

Initial status: UNTESTED_HYPOTHESIS.
Candidate results will appear in the evidence ledger with their exact evidence class.
No modeled, replayed, or component result is a LIVE_E2E architecture gain.

## 17. Both-size/both-mode matrix

| Cell | Status |
|---|---|
| 1.4B fixed/adaptive | Targeted, candidate quality and performance NOT_RUN |
| 2.6B fixed/adaptive | Targeted, tiny runtime sanity only |
| P2/P4 | Physical candidate measurement pending |
| P8 | Projection only |

## 18. Loop-specificity control

Retain all operations and KV identities in an aliased logical unroll.
Compare a conventional executor with the same aliases and ownership.
A benefit that survives without recurrence belongs to the generic category.

## 19. Bounded implementation

Do not build a serving engine before the smallest test supports this mechanism.
{c['next']}

## 20. Decision and next action

No finalist decision is available at this freeze.
The single next action is the falsification test in Section 15.
'''
        path=ROOT/'candidates'/f'{c["id"]}.md'
        assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    report=['# Initial candidate register','','This register freezes mechanisms before candidate selection. It does not claim that fourteen architectures are novel.','',
        '| ID | Mechanism | Family | Initial category |','|---|---|---|---|']
    report += [f'| {c["id"]} | {c["title"]} | {c["family"]} | {c["axis"]} |' for c in CARDS]
    report += ['','C09, C10, and C11 originated outside seed areas A–F. Their generic overlap remains an explicit risk.',
        'Candidate counts do not establish the required effort allocation or ten-hour duration.',
        'The time ledger must establish actual activity.','']
    (ROOT/'reports/candidate_register.md').write_text('\n'.join(report))
    print('FROZEN',len(CARDS),'distinct mechanism cards. No positive result asserted.')

if __name__=='__main__':main()
