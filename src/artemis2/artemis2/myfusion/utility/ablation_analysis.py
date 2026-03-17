import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

def get_results():
    results = list()
    results_bilstm = list()
    results_gru = list()
    results_conv = list()
    titles = list()
    with (open('results.txt', 'r') as f):
        lines = f.readlines()
        i = 0
        for line in lines:
            if line.startswith('%'):
                i = 0
                text = line.replace('%', '').replace('\n', '').strip()
                text = text.replace('POSITIONAL ENCODER RESIDUAL', 'R_Hv') \
                    .replace('SUMMARY RESIDUAL', 'R_s') \
                    .replace('BACKBONE RESIDUAL', 'R_b') \
                    .replace('LINEAR2 RESIDUAL', 'R_Hq') \
                    .replace('IMAGE RESIDUAL', 'F_r') \
                    .replace(' NETWORK', '')
                titles.append(text)
            else:
                value = line.split('&')[-1].strip()
                try:
                    value = float(value.replace('\\', ''))
                except ValueError:
                    value = np.mean(results)
                if i == 0:
                    results.append(value)
                elif i == 1:
                    results_bilstm.append(value)
                elif i == 2:
                    results_gru.append(value)
                elif i == 3:
                    results_conv.append(value)
                i += 1
    return results, results_bilstm, results_gru, results_conv, titles


def plot_ablation(results, results_bilstm, results_gru, results_conv, titles):
    plt.figure(figsize=(10, 6))
    plt.plot(results, label='No text processing')
    plt.plot(results_bilstm, label='biLSTM')
    plt.plot(results_gru, label='GRU')
    plt.plot(results_conv, label='Conv')
    plt.xticks(range(len(titles)), titles, rotation=90)
    plt.xlabel('Network configurations')
    plt.ylabel('mAP')
    plt.legend()
    plt.tight_layout()
    #plt.title('Ablation results')
    plt.show()


def compare_results(results, results_bilstm, results_gru, results_conv):
    # Converting lists to numpy arrays for easy manipulation
    results_np = np.array(results)
    results_bilstm_np = np.array(results_bilstm)
    results_gru_np = np.array(results_gru)
    results_conv_np = np.array(results_conv)

    # Compute basic statistics for each list
    def print_stats(name, values):
        print(f"Statistics for {name}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std Dev: {np.std(values):.4f}")
        print(f"  Min: {np.min(values):.4f}")
        print(f"  Max: {np.max(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print()

    print_stats("No text processing", results_np)
    print_stats("biLSTM", results_bilstm_np)
    print_stats("GRU", results_gru_np)
    print_stats("Conv", results_conv_np)

    # Visual comparison: Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot([results_np, results_bilstm_np, results_gru_np, results_conv_np],
                labels=['No text processing', 'biLSTM', 'GRU', 'Conv'])
    plt.ylabel('mAP')
    plt.title('Comparison of Model Performance')
    plt.grid(True)
    plt.savefig('boxplot_ablation.pdf')

    combinations = {
        'No text processing vs biLSTM': (results_np, results_bilstm_np),
        'No text processing vs GRU': (results_np, results_gru_np),
        'No text processing vs Conv': (results_np, results_conv_np),
        'biLSTM vs GRU': (results_bilstm_np, results_gru_np),
        'biLSTM vs Conv': (results_bilstm_np, results_conv_np),
        'GRU vs Conv': (results_gru_np, results_conv_np)
    }

    # Paired t-test for all combinations
    print("Paired t-test comparisons:")
    for name, (data1, data2) in combinations.items():
        t_stat, p_value = ttest_rel(data1, data2)
        print(f"{name}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # Wilcoxon Signed-Rank Test for all combinations
    print("\nWilcoxon Signed-Rank Test comparisons:")
    for name, (data1, data2) in combinations.items():
        try:
            wilcox = wilcoxon(data1, data2)
            print(f"{name}: stat = {wilcox.statistic:.4f}, p-value = {wilcox.pvalue:.4f}")
        except ValueError as e:
            print(f"{name}: Wilcoxon test could not be performed ({e})")

    """# Paired t-test between models to see if one consistently outperforms another
    print("Paired t-test comparisons:")
    _, p_value_bilstm = ttest_rel(results_np, results_bilstm_np)
    print(f"No text processing vs biLSTM: p-value = {p_value_bilstm:.4f}")

    _, p_value_gru = ttest_rel(results_np, results_gru_np)
    print(f"No text processing vs GRU: p-value = {p_value_gru:.4f}")

    _, p_value_conv = ttest_rel(results_np, results_conv_np)
    print(f"No text processing vs Conv: p-value = {p_value_conv:.4f}")

    # If t-test assumptions don't hold (e.g., normal distribution), use Wilcoxon test
    print("\nWilcoxon Signed-Rank Test comparisons:")
    wilcox_bilstm = wilcoxon(results_np, results_bilstm_np)
    print(f"No text processing vs biLSTM: stat = {wilcox_bilstm.statistic:.4f}, p-value = {wilcox_bilstm.pvalue:.4f}")

    wilcox_gru = wilcoxon(results_np, results_gru_np)
    print(f"No text processing vs GRU: stat = {wilcox_gru.statistic:.4f}, p-value = {wilcox_gru.pvalue:.4f}")

    wilcox_conv = wilcoxon(results_np, results_conv_np)
    print(f"No text processing vs Conv: stat = {wilcox_conv.statistic:.4f}, p-value = {wilcox_conv.pvalue:.4f}")"""


if __name__ == '__main__':
    results, results_bilstm, results_gru, results_conv, titles = get_results()
    #plot_ablation(results, results_bilstm, results_gru, results_conv, titles)
    compare_results(results, results_bilstm, results_gru, results_conv)
