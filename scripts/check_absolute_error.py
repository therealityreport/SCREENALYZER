"""Check absolute error vs 4s target."""

# Ground truth
GT = {
    'BRANDI': 10014,
    'EILEEN': 10001,
    'KIM': 48004,
    'KYLE': 21017,
    'LVP': 2018,
    'RINNA': 25015,
    'YOLANDA': 16002
}

# Auto from latest run
AUTO = {
    'BRANDI': 5835,
    'EILEEN': 16583,
    'KIM': 48751,
    'KYLE': 23168,
    'LVP': 2417,
    'RINNA': 20833,
    'YOLANDA': 8750
}

print('Absolute Error Analysis (Target: ≤4000ms):')
print()
print('Person      Auto (ms)  GT (ms)   Δ (ms)   Abs Δ    Status')
print('-' * 65)

within_target = 0
for person in sorted(GT.keys()):
    auto = AUTO[person]
    gt = GT[person]
    delta = auto - gt
    abs_delta = abs(delta)
    status = '✓' if abs_delta <= 4000 else '✗'
    if abs_delta <= 4000:
        within_target += 1

    print(f'{person:<10} {auto:>8} {gt:>8} {delta:>+10} {abs_delta:>10}   {status}')

print('-' * 65)
print(f'Within ≤4s target: {within_target}/7')
print()

# List those above target
above_target = [(p, abs(AUTO[p] - GT[p])) for p in GT if abs(AUTO[p] - GT[p]) > 4000]
if above_target:
    print('Above ≤4s target:')
    for person, abs_err in sorted(above_target, key=lambda x: -x[1]):
        print(f'  {person}: {abs_err}ms over')
