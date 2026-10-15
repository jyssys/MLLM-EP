"""Conservative final-answer extraction. Never execute generated code."""
import re


def final_answer(text):
    answers=[]
    for match in re.finditer(r'\\boxed\s*\{',text):
        start=match.end();depth=1;end=start
        while end<len(text) and depth:
            depth += (text[end]=='{')-(text[end]=='}');end+=1
        if depth==0:answers.append(text[start:end-1].strip())
    if answers:return answers[-1],'LAST_BALANCED_BOX'
    tail=text.rsplit('</think>',1)[-1][-4000:]
    matches=list(re.finditer(r'(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?([^\n$.]{1,100})',tail,re.I))
    return (matches[-1].group(1).strip(),'EXPLICIT_FINAL_MARKER') if matches else (None,'UNPARSEABLE')


def aime_integer(value):
    if value is None:return None
    value=re.sub(r'\\(?:text|mathrm)\{(\d+)\}',r'\1',value).strip(' ${}\\')
    if not re.fullmatch(r'\d{1,3}',value):return None
    return int(value)
