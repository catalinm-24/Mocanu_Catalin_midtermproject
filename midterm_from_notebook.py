# %% Cell 1
import pandas as pd
import os

def load_dataset(dataset_name):
    path = f"data/{dataset_name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {dataset_name}.csv - {len(df)} rows")
    return df

# %% Cell 2
#Step 2.1 - Count item combinations of size k
from itertools import combinations
from collections import Counter

def count_itemsets(transactions, k:int) -> Counter:
    counts = Counter()
    for t in transactions:
        if len(t) < k:
            continue
        for comb in combinations(sorted(t),k):
            counts[comb] += 1
    return counts

# %% Cell 3
#Step 2.2 - Compute support and keep frequent items

from collections import Counter
from itertools import combinations

def get_frequent_itemsets(transactions, min_support=0.2):
    n = len(transactions)
    supports = {}
    frequent = {}
    k = 1

    while True:
        counts = count_itemsets(transactions, k)
        freq_k = []
        for itemset, cnt in counts.items():
            supp = cnt / n
            if supp >= min_support:
                freq_k.append(itemset)
                supports[itemset] = supp
        if not freq_k:
            break
        frequent[k] = sorted(freq_k, key=lambda s: (-supports[s],s))
        k += 1
    return frequent, supports

# %% Cell 4
#Step 2.3 - Generate rules (A->B) from frequent itemsets
from itertools import combinations

def support_fraction(itemset, transactions):
    n = len(transactions)
    if n == 0:
        return 0.0
    iset = set(itemset)
    return sum(1 for t in transactions if iset.issubset(t)) / n

def generate_rules(frequent, supports, transactions, min_conf: float=0.6):
    rules = []
    for k, itemsets in frequent.items():
        if k < 2:
            continue
        for L in itemsets:
            L = tuple(sorted(L))
            supp_L = supports.get(L, support_fraction(L, transactions))
            items = list(L)

            for r in range(1, len(items)):
                for A in combinations(items, r):
                    A = tuple(sorted(A))
                    B = tuple(sorted(set(items) - set(A)))

                    supp_A = supports.get(A)
                    if supp_A is None:
                        supp_A = support_fraction(A, transactions)
                    if supp_A == 0:
                        continue  

                    conf = supp_L / supp_A
                    if conf >= min_conf:
                        supp_B = supports.get(B)
                        if supp_B is None:
                            supp_B = support_fraction(B, transactions)
                        lift = (conf / supp_B) if supp_B > 0 else None
                        rules.append((A, B, supp_L, conf, lift))
                        
    rules.sort(key=lambda x: (-x[3], -x[2], x[0], x[1]))
    return rules
    

# %% Cell 5
# Step 2.4 — Brute Force helpers (definitions only; called by the menu)

import os
import pandas as pd

def run_bruteforce_for_dataset(dataset_name: str, min_support: float, min_conf: float, save: bool = True):
    tx = load_transactions_csv(dataset_name)

    frequent, supports = get_frequent_itemsets(tx, min_support=min_support)
    rules = generate_rules(frequent, supports, tx, min_conf=min_conf)

    if save:
        os.makedirs("output", exist_ok=True)
        # frequent itemsets file
        freq_rows = []
        for k, itemsets in frequent.items():
            for it in itemsets:
                freq_rows.append([k, ",".join(it), supports[it]])
        pd.DataFrame(freq_rows, columns=["k", "itemset", "support"]).to_csv(
            f"output/{dataset_name}_frequent_itemsets.csv", index=False
        )
        # rules file
        rule_rows = [
            [",".join(A), ",".join(B), supp, conf, (None if lift is None else float(f"{lift:.6f}"))]
            for A, B, supp, conf, lift in rules
        ]
        pd.DataFrame(rule_rows, columns=["antecedent","consequent","support","confidence","lift"]).to_csv(
            f"output/{dataset_name}_rules.csv", index=False
        )

    total_itemsets = sum(len(v) for v in frequent.values())
    summary_row = {
        "Dataset": dataset_name.upper(),
        "#Transactions": len(tx),
        "#Frequent Itemsets": total_itemsets,
        "#Rules": len(rules),
        "Time (s)": None  
    }
    return summary_row, rules


def run_bruteforce_all(datasets, min_support: float, min_conf: float, save: bool = True):
    rows = []
    all_rules = {} 
    for ds in datasets:
        summary_row, rules = run_bruteforce_for_dataset(ds, min_support=min_support, min_conf=min_conf, save=save)
        rows.append(summary_row)
        all_rules[ds] = rules
    summary_df = pd.DataFrame(rows, columns=["Dataset","#Transactions","#Frequent Itemsets","#Rules","Time (s)"])
    return summary_df, all_rules


# %% Cell 6
# Step 3.1 – Data preparation helper functions for Apriori & FP-Growth

import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def load_transactions_csv(dataset_name):
    path = f"data/{dataset_name}.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []

    df = pd.read_csv(path)
    if "Transaction" not in df.columns:
        print(f"'Transaction' column missing in {dataset_name}.csv")
        print("Columns found:", df.columns.tolist())
        return []

    transactions = (
        df["Transaction"]
        .astype(str)
        .str.split(",")
        .apply(lambda L: [s.strip().lower() for s in L if s.strip()])
        .tolist()
    )
    return transactions


def one_hot_encode(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


# %% Cell 7
#Step 3.2 Apriori definition

def run_apriori_for_dataset(dataset_name: str, min_support: float, min_conf: float, save: bool = True):
    tx = load_transactions_csv(dataset_name)
    df_enc = one_hot_encode(tx)
    freq_ap = apriori(df_enc, min_support=min_support, use_colnames=True)
    rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_conf)
    if save:
        os.makedirs("output", exist_ok=True)
        freq_ap.to_csv(f"output/{dataset_name}_apriori_frequent_itemsets.csv", index=False)
        rules_ap.to_csv(f"output/{dataset_name}_apriori_rules.csv", index=False)
    return {
        "Dataset": dataset_name.upper(),
        "#Transactions": len(tx),
        "#Frequent Itemsets": len(freq_ap),
        "#Rules": len(rules_ap)
    }, rules_ap


# %% Cell 8
#Step 3.3 Run FP-Growth on one dataset

def run_fpgrowth_for_dataset(dataset_name: str, min_support: float, min_conf: float, save: bool = True):
    tx = load_transactions_csv(dataset_name)
    df_enc = one_hot_encode(tx)
    freq_fp = fpgrowth(df_enc, min_support=min_support, use_colnames=True)
    rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=min_conf)
    if save:
        os.makedirs("output", exist_ok=True)
        freq_fp.to_csv(f"output/{dataset_name}_fpgrowth_frequent_itemsets.csv", index=False)
        rules_fp.to_csv(f"output/{dataset_name}_fpgrowth_rules.csv", index=False)
    return {
        "Dataset": dataset_name.upper(),
        "#Transactions": len(tx),
        "#Frequent Itemsets": len(freq_fp),
        "#Rules": len(rules_fp)
    }, rules_fp


# %% Cell 9
# Step 4 — Unified Menu Runner (single interactive cell)

import os, time, pandas as pd

# ---- Input prompts with validation ----
print("=== Association Rule Mining Menu ===")
DATASETS = ["amazon", "bestbuy", "sephora", "target", "ikea"]

ds_in = input("Dataset (amazon/bestbuy/sephora/target/ikea/all): ").strip().lower()
while ds_in not in DATASETS + ["all"]:
    ds_in = input("Please enter one of amazon/bestbuy/sephora/target/ikea/all: ").strip().lower()

algo_in = input("Algorithm (brute/apriori/fpgrowth/both/all): ").strip().lower()
while algo_in not in ["brute", "apriori", "fpgrowth", "both", "all"]:
    algo_in = input("Please enter one of brute/apriori/fpgrowth/both/all: ").strip().lower()

def ask_float(prompt, lo=0.0, hi=1.0, default=None):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            v = float(s)
            if lo <= v <= hi:
                return v
        except:
            pass
        print(f"Enter a number in [{lo}, {hi}]")

min_support = ask_float("Minimum support (0–1) [default 0.2]: ", 0.0, 1.0, default=0.2)
min_conf    = ask_float("Minimum confidence (0–1) [default 0.6]: ", 0.0, 1.0, default=0.6)

datasets = DATASETS if ds_in == "all" else [ds_in]
os.makedirs("output", exist_ok=True)

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round(time.perf_counter() - t0, 6)

rows = []

for ds in datasets:
    print("\n" + "="*60)
    print(f"Processing {ds.upper()} (minsup={min_support}, minconf={min_conf})")
    print("="*60)

    try:
        tx = load_transactions_csv(ds)          
        df_enc = one_hot_encode(tx)               

        # Decide which to run
        run_brute   = algo_in in ("brute", "all", "both")  # 'both' runs Apriori+FP; brute is included if 'all' is selected.
        run_ap      = algo_in in ("apriori", "both", "all")
        run_fp      = algo_in in ("fpgrowth", "both", "all")

        # If 'both', exclude brute (both = Apriori + FP); if 'all', include all 3.
        if algo_in == "both":
            run_brute = False
        if algo_in == "brute":
            run_ap = run_fp = False

        # ---- Brute Force (custom) ----
        if run_brute:
            def _brute():
                frequent, supports = get_frequent_itemsets(tx, min_support=min_support)
                rules = generate_rules(frequent, supports, tx, min_conf=min_conf)
                freq_rows = []
                for k, itemsets in frequent.items():
                    for it in itemsets:
                        freq_rows.append([k, ",".join(it), supports[it]])
                pd.DataFrame(freq_rows, columns=["k","itemset","support"]).to_csv(
                    f"output/{ds}_frequent_itemsets.csv", index=False
                )
                rule_rows = [
                    [",".join(A), ",".join(B), s, c, (None if L is None else float(f"{L:.6f}"))]
                    for A,B,s,c,L in rules
                ]
                pd.DataFrame(rule_rows, columns=["antecedent","consequent","support","confidence","lift"]).to_csv(
                    f"output/{ds}_bruteforce_rules.csv", index=False
                )
                return sum(len(v) for v in frequent.values()), len(rules)

            (bf_itemsets, bf_rules), t = timed(_brute)
            rows.append([ds.upper(), "Brute Force", len(tx), bf_itemsets, bf_rules, t])
            print(f"Brute Force → itemsets: {bf_itemsets}, rules: {bf_rules}, time={t:.4f}s")
            print(f" output/{ds}_frequent_itemsets.csv")
            print(f" output/{ds}_bruteforce_rules.csv")

        # ---- Apriori (mlxtend) ----
        if run_ap:
            def _ap():
                freq_ap = apriori(df_enc, min_support=min_support, use_colnames=True)
                rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_conf)
                freq_ap.to_csv(f"output/{ds}_apriori_frequent_itemsets.csv", index=False)
                rules_ap.to_csv(f"output/{ds}_apriori_rules.csv", index=False)
                return len(freq_ap), len(rules_ap)
            (ap_itemsets, ap_rules), t = timed(_ap)
            rows.append([ds.upper(), "Apriori", len(tx), ap_itemsets, ap_rules, t])
            print(f"Apriori     → itemsets: {ap_itemsets}, rules: {ap_rules}, time={t:.4f}s")
            print(f" output/{ds}_apriori_frequent_itemsets.csv")
            print(f" output/{ds}_apriori_rules.csv")

        # ---- FP-Growth (mlxtend) ----
        if run_fp:
            def _fp():
                freq_fp = fpgrowth(df_enc, min_support=min_support, use_colnames=True)
                rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=min_conf)
                freq_fp.to_csv(f"output/{ds}_fpgrowth_frequent_itemsets.csv", index=False)
                rules_fp.to_csv(f"output/{ds}_fpgrowth_rules.csv", index=False)
                return len(freq_fp), len(rules_fp)
            (fp_itemsets, fp_rules), t = timed(_fp)
            rows.append([ds.upper(), "FP-Growth", len(tx), fp_itemsets, fp_rules, t])
            print(f"FP-Growth   → itemsets: {fp_itemsets}, rules: {fp_rules}, time={t:.4f}s")
            print(f" output/{ds}_fpgrowth_frequent_itemsets.csv")
            print(f" output/{ds}_fpgrowth_rules.csv")

    except Exception as e:
        print(f"Error on {ds}: {type(e).__name__}: {e}")
        rows.append([ds.upper(), algo_in.title(), None, None, None, None])
        continue

# ---- Summary table & save ----
summary = pd.DataFrame(rows, columns=[
    "Dataset","Algorithm","#Transactions","#Frequent Itemsets","#Rules","Time (s)"
])
print("\n=== Summary ===")
print(summary)
summary.to_csv("output/menu_run_summary.csv", index=False)
print("Saved: output/menu_run_summary.csv")


# %% Cell 10