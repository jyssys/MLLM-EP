"""CPU controls for composition depth versus state-evaluation dependency."""
import argparse
import datetime as dt
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha, write


@dataclass(frozen=True)
class Affine:
    matrix: tuple
    bias: tuple

    def __post_init__(self):
        n = len(self.bias)
        if not n or len(self.matrix) != n or any(len(row) != n for row in self.matrix):
            raise ValueError('Require a nonempty square affine map.')

    def apply(self, value):
        if len(value) != len(self.bias):
            raise ValueError('Input dimension differs from the affine map.')
        return tuple(sum(a*x for a, x in zip(row, value))+b
                     for row, b in zip(self.matrix, self.bias))


def compose(after, before):
    """Return after(before(x)). This performs arithmetic, not pointer composition."""
    n = len(after.bias)
    if len(before.bias) != n:
        raise ValueError('Affine dimensions differ.')
    matrix = tuple(tuple(sum(after.matrix[i][k]*before.matrix[k][j] for k in range(n))
                         for j in range(n)) for i in range(n))
    return Affine(matrix, after.apply(before.bias))


def balanced_affine(maps):
    if not maps:
        raise ValueError('Require at least one map.')
    if len(maps) == 1:
        return maps[0], 0
    middle = len(maps)//2
    left, ld = balanced_affine(maps[:middle])
    right, rd = balanced_affine(maps[middle:])
    return compose(right, left), max(ld, rd)+1


def balanced_opaque(indices, events):
    """Construct a composition tree. Evaluation must still produce each input."""
    if not indices:
        raise ValueError('Require at least one leaf.')
    if len(indices) == 1:
        index = indices[0]

        def leaf(value):
            result = value*value+index
            events.append(dict(leaf=index, input=value, output=result))
            return result

        return leaf, 0
    middle = len(indices)//2
    left, ld = balanced_opaque(indices[:middle], events)
    right, rd = balanced_opaque(indices[middle:], events)
    return lambda value: right(left(value)), max(ld, rd)+1


def naive_affine_counts(n, rounds):
    if n < 1 or rounds < 1:
        raise ValueError('Require positive dimensions and rounds.')
    # Conventional dense dot products, with no fused arithmetic or zero pruning.
    apply_mul, apply_add = n*n, n*n
    compose_mul, compose_add = n**3+n*n, n*n*(n-1)+n*n
    return dict(serial_multiplications=rounds*apply_mul,
                serial_additions=rounds*apply_add,
                folded_multiplications=(rounds-1)*compose_mul+apply_mul,
                folded_additions=(rounds-1)*compose_add+apply_add,
                scope='Scalar-operation counts for this dense affine construction. No latency prediction.')


def exhaustive_table_log2_bytes(width, bits=16):
    """Size of this explicit all-bit-pattern table, not a representation bound."""
    if width < 1 or bits < 1 or bits % 8:
        raise ValueError('Require positive width and a whole-byte word size.')
    row_bytes = width*(bits//8)
    if row_bytes & (row_bytes-1):
        raise ValueError('This exact integer exponent requires a power-of-two row size.')
    return bits*width+row_bytes.bit_length()-1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    maps = [Affine(((Fraction(i+1), Fraction(1, 2)),
                    (Fraction(1, 3), Fraction(i+2))), (Fraction(i), Fraction(-i)))
            for i in range(1, 5)]
    initial = (Fraction(2, 3), Fraction(-3, 2))
    serial = initial
    for item in maps:
        serial = item.apply(serial)
    folded, build_depth = balanced_affine(maps)
    assert folded.apply(initial) == serial
    events = []
    opaque, opaque_build_depth = balanced_opaque([1, 2, 3, 4], events)
    opaque_output = opaque(1)
    assert len(events) == 4 and all(a['output'] == b['input'] for a, b in zip(events, events[1:]))
    native = Path(read(ROOT/'artifacts/semantics_manifest.json')['runtime']['native_vllm_source_path'])
    write(target, dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        evidence_class='CPU_CONSTRUCTED_CONTROL', candidate_effect='NO_ADDITIONAL_CARD_OR_PROMOTION',
        affine_positive_control=dict(exact_rational_parity=True, rounds=4, width=2,
            composition_tree_levels=build_depth, output=[str(x) for x in serial],
            counts=naive_affine_counts(2, 4)),
        opaque_control=dict(rounds=4, composition_tree_levels=opaque_build_depth,
            ordered_state_evaluations=len(events), output=opaque_output, events=events,
            scope='This callback representation retains the serial input chain. Not a lower bound for every algorithm.'),
        polynomial_example=dict(single_degree=2, degree_after_four_compositions=2**4,
            scope='The x-squared example is not a fixed-degree closed family. It is not an Ouro equivalence proof.'),
        exhaustive_representation=dict(width=2048, input_word_bits=16, output_bytes=4096,
            log2_total_bytes=exhaustive_table_log2_bytes(2048),
            scope='One full bit-pattern lookup representation, excluding history. Not a bound on all representations.'),
        native_source=dict(path=str(native), sha256=sha(native),
            read_extents=['78-105', '165-211', '262-285', '348-378'],
            observations=['SiluAndMul follows the merged projection.',
                          'Attention selects a depth-indexed cache object.',
                          'Layer execution preserves normalization and residual updates.',
                          'Each recurrent pass consumes the previous normalized output.']),
        source_sha256=sha(Path(__file__)),
        transfer_status='NO_QUALIFIED_COMPACT_COMPOSITION_FOR_UNCHANGED_OURO',
        gpu_execution='NOT_RUN', numerical_model_parity='NOT_RUN', latency='NOT_RUN',
        limits=['Affine rational closure does not establish BF16 reassociation parity.',
                'Pointer composition does not precompute nonlinear state transitions.',
                'No impossibility claim excludes a different useful representation.',
                'This auxiliary screen changes no candidate verdict.']))


if __name__ == '__main__':
    main()
