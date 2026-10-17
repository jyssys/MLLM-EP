"""Frozen final-answer extraction with separate parsing and completion status."""
import re
from decimal import Decimal, InvalidOperation
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer
PARSER_VERSION = 'parpp-final-v2'


def numeric(value):
    if value is None:
        return None
    value = value.replace(',', '').replace('$','').replace('\\!','').strip()
    value = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r'\1', value)
    value = value.strip(' {}')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', value):
        return None
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return None


def score(row, result):
    extracted, rule = final_answer(result['text'])
    if rule=='EXPLICIT_FINAL_MARKER':
        # The inherited integer-task fallback stopped at every period.
        # Retain decimal points and complete delimited expressions instead.
        tail=result['text'].rsplit('</think>',1)[-1][-4000:]
        markers=list(re.finditer(r'(?:final\s+answer|answer)\s*(?:is|:|=)\s*',tail,re.I))
        if markers:
            line=tail[markers[-1].end():].split('\n',1)[0].replace('<|im_end|>','').strip()
            math=re.match(r'^\$([^$\n]+)\$',line)
            extracted=math.group(1).strip() if math else line.removesuffix('.').strip()
            rule='EXPLICIT_FINAL_LINE_V2'
    if extracted is None and row['dataset']=='gsm8k':
        matches = re.findall(r'####\s*([+-]?[\d,]+(?:\.\d+)?)', result['text'])
        if matches:
            extracted, rule = matches[-1], 'EXPLICIT_GSM_MARKER'
    if extracted is None:
        tail=result['text'].rsplit('</think>',1)[-1].replace('<|im_end|>','').strip()
        if numeric(tail) is not None:
            extracted,rule=tail,'STANDALONE_FINAL_NUMBER'
    if row['dataset']=='gsm8k':
        prediction = numeric(extracted)
        correct = prediction is not None and prediction == numeric(row['answer'])
        parsed = prediction is not None
    else:
        prediction = extracted
        parsed, correct = False, False
        if extracted is not None:
            try:
                import sys
                sys.path.insert(0,'/home/esjung/external/looped-exact-depth-chunk-assets/cpu_deps')
                from math_verify import parse, verify, LatexExtractionConfig
                gold=parse('$'+row['answer']+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                guess=parse('$'+extracted+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                parsed=bool(guess)
                correct=bool(verify(gold,guess,strict=True,timeout_seconds=2))
            except Exception:
                pass
    complete = result['finish_reason']=='eos'
    return dict(extracted_answer=prediction,parse_rule=rule,parsed=parsed,
        answer_matches=bool(correct),correct_completed=bool(correct and complete),
        truncated=not complete,reference_answer=row['answer'],parser_version=PARSER_VERSION)
