import json

qb = json.load(open('data/question_bank.json'))
qs = qb['questions']
assert len(qs) >= 8, f"Expected >=8 questions, got {len(qs)}"
critical = {'oxygen_saturation','heart_rate','respiratory_rate','systolic_bp','chest_pain','pain_radiation','shortness_of_breath','confusion'}
covered = {q['field'] for q in qs}
missing = critical - covered
assert not missing, f"Missing critical fields: {missing}"
required_keys = ['id','question','field','category','risk_impact','uncertainty_reduction','applicable_to']
for q in qs:
    for k in required_keys:
        qid = q.get('id', 'unknown')
        assert k in q, f"Missing key '{k}' in question '{qid}'"
print(f"question_bank OK: {len(qs)} questions, all {len(critical)} critical fields covered")

sc = json.load(open('data/synthetic_cases.json'))
cases = {c['id']: c for c in sc['cases']}
for expected_id in ['high_risk_chest_pain', 'contradictory_breathing_case', 'low_risk_muscle_pain']:
    assert expected_id in cases, f"Missing case: {expected_id}"
for cid, c in cases.items():
    steps = c['steps']
    orders = [s['order'] for s in steps]
    assert orders == sorted(orders), f"Steps not sorted in: {cid}"
    for s in steps:
        for k in ['order', 'field', 'value', 'label', 'clinical_note']:
            assert k in s, f"Missing '{k}' in step of {cid}"
    assert 'expected_routing' in c, f"No expected_routing in {cid}"
print(f"synthetic_cases OK: {len(cases)} cases, all schema keys present, steps ordered correctly")
