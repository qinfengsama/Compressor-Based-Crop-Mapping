import subprocess

scripts = 'ours.py'

datasets = ["Pastis", "German", "France"]
periods = ['43', '36', '22']
comps = ['gzip', 'zstandard', 'bz2']
shots = ['50', '20', '10', '5']
seeds = ['32', '47', '2024', '400', '21']


def run(scripts, dataset, is_analysis, period, comp, seed, code, alphabet_len, train_num, k):
    if is_analysis:
        command = [
                'python', scripts,
                '--dataset', dataset,
                '--is_analysis', 
                '--period', period,
                '--compressor', comp,
                '--seed', seed, 
                '--code', code,
                '--alphabet_len', alphabet_len,
                '--train_num', train_num,
                '--k', k,
            ]
        subprocess.run(command)
    else:
        command = [
                'python', scripts,
                '--dataset', dataset,
                # '--is_analysis', 
                '--period', period,
                '--compressor', comp,
                '--seed', seed, 
                '--code', code,
                '--alphabet_len', alphabet_len,
                '--train_num', train_num,
                '--k', k,
            ]
        subprocess.run(command)


# ---------------------------------- Section 3.1 Comparisons with Deep Learning Models ----------------------------------
for d, pp in zip(datasets, periods):
    run(scripts=scripts, dataset=d, is_analysis=False, period=pp, comp='gzip', seed='32', code='char', alphabet_len='22', train_num='0.2', k='2')


# ---------------------------------- Section 3.2 Few-Shot Learning ----------------------------------
for d, pp in zip(datasets, periods):
    for shot in shots:
        for seed in seeds:
            run(scripts=scripts, dataset=d, is_analysis=False, period=pp, comp='gzip', seed=seed, code='char', alphabet_len='22', train_num=shot, k='2')


# ---------------------------------- Section 4.1 Different Alphabet Lengths ----------------------------------
# Pure numerical mapping
for d, pp in zip(datasets, periods):
    for seed in seeds:
        run(scripts=scripts, dataset=d, is_analysis=True, period=pp, comp='gzip', seed=seed, code='num', alphabet_len='22', train_num='0.1', k='2')

# Symbolic representation mapping under different alphabet lengths
for d, pp in zip(datasets, periods):
    for al in range(2, 53, 5):
        for seed in seeds:
            run(scripts=scripts, dataset=d, is_analysis=True, period=pp, comp='gzip', seed=seed, code='char', alphabet_len=str(al), train_num='0.1', k='2')


# ---------------------------------- Section 4.2 Different Compressors ----------------------------------
for d, pp in zip(datasets, periods):
    for comp in comps:
        for seed in seeds:
            run(scripts=scripts, dataset=d, is_analysis=True, period=pp, comp=comp, seed=seed, code='char', alphabet_len='22', train_num='0.1', k='2')


