import numpy as np


def rank():
    results = list()
    with open('results.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('%'):
                continue
            value = line.split('&')[-1].strip()
            if value.startswith('\\textbf'):
                value = value[8:-3]
            value = float(value.replace('\\', ''))
            results.append(value)
    values, counts = np.unique(results, return_counts=True)
    for v, c in zip(values[::-1], counts[::-1]):
        print(f'{v}: count={c}')


def find(v: float):
    with open('results.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('%'):
                continue
            value = line.split('&')[-1].strip()
            if value.startswith('\\textbf'):
                value = value[8:-3]
            value = float(value.replace('\\', ''))
            if value == v:
                info = [l.strip() for l in line.split('&')]
                print(f'{v}: {"residual" if info[4] == "True" else ""} '
                      f'{"sumresidual" if info[5] == "True" else ""} '
                      f'{"backbone" if info[6] == "True" else ""} '
                      f'{"linear2" if info[7] == "True" else ""} '
                      f'{"imageresidual" if info[8] == "True" else ""} '
                      f'{info[3]}')


if __name__ == "__main__":
    #rank()
    find(76.57)
