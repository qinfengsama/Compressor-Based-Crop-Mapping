import subprocess

scripts = '0412_jstars.py'

areas = ["t30uxv", "t31tfj", "t31tfm", "t32ulu"]
periods = ['43', '61', '46', '38']
alphabet_lens = '45'
k = '4'
comps = ['gzip', 'zstandard', 'bz2']
rates = ['50', '20', '10', '5', '1']


def run(scripts, area, period, comp, code, alphabet_len, train_num, k):
    command = [
            'python', scripts,
            '--area', area,
            '--period', period,
            '--compressor', comp,
            '--code', code,
            '--alphabet_len', alphabet_len,
            '--train_num', train_num,
            '--k', k,
        ]
    subprocess.run(command)
    print(f"Completed! Waiting 2 seconds...")


# ---------------------------------- Forward ----------------------------------
# for a, pp in zip(areas, periods):
#     # few_shot
#     for r in rates:
#         run(scripts, a, pp, 'gzip', 'char', alphabet_lens, r, k)
#
#     # different periods
#     if a == "t30uxv":
#         ps = ["43", "30", "21", "10"]
#     elif a == "t31tfj":
#         ps = ["61", "45", "30", "15"]
#     elif a == "t31tfm" or a == "t31tfm_1":
#         ps = ["46", "33", "23", "11"]
#     else:
#         ps = ["38", "27", "19", "9"]
#
#     for p in ps:
#         run(scripts, a, p, 'gzip', 'char', alphabet_lens, '0.2', k)
#
#     # num
#     run(scripts, a, pp, 'gzip', 'num', alphabet_lens, '0.2', k)
#
#     # different compressors
#     for c in comps:
#         run(scripts, a, pp, c, 'char', alphabet_lens, '0.2', k)


for a, pp in zip(areas, periods):
    # different alphabet_len
    for al in range(3, 53):
        run(scripts, a, pp, 'gzip', 'char', str(al), '0.2', k)

