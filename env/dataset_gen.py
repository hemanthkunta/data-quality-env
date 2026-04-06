from __future__ import annotations

import numpy as np
import pandas as pd

NULL_DISGUISES = ["NULL", "N/A", "UNKNOWN", "-", "", "0", "none"]


def generate_dataset(task_id: int, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Returns:
      tables_dict: {table_name: DataFrame}
      gold_faults: dict
    """
    rng = np.random.default_rng(seed)
    if task_id == 1:
        return _task1(rng, seed)
    if task_id == 2:
        return _task2(rng)
    if task_id == 3:
        return _task3(rng)
    if task_id == 4:
        return _task4(rng)
    raise ValueError(f"Unknown task_id {task_id}")


def _task1(rng: np.random.Generator, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    n = 200
    df = pd.DataFrame(
        {
            "customer_id": range(1001, 1001 + n),
            "email": [f"user{i}@example.com" for i in range(n)],
            "name": [f"Name {i}" for i in range(n)],
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "country": rng.choice(["US", "UK", "IN", "DE", "FR"], n).tolist(),
        }
    )

    real_null_cid = int(rng.integers(3, 7))
    null_cid_idx = rng.choice(n, real_null_cid, replace=False)
    df.loc[null_cid_idx, "customer_id"] = None

    real_null_email = int(rng.integers(8, 15))
    null_email_idx = rng.choice(n, real_null_email, replace=False)
    df.loc[null_email_idx, "email"] = None

    disguised_null_email = int(rng.integers(4, 9))
    avail = [i for i in range(n) if i not in set(null_email_idx.tolist())]
    dis_idx = rng.choice(avail, disguised_null_email, replace=False)
    df.loc[dis_idx, "email"] = rng.choice(NULL_DISGUISES, disguised_null_email).tolist()

    dup_count = int(rng.integers(10, 19))
    dup_src = rng.choice(n, dup_count, replace=True)
    dups = df.iloc[dup_src].copy()
    df = pd.concat([df, dups], ignore_index=True)

    near_dup_count = int(rng.integers(5, 9))
    near_src = rng.choice(n, near_dup_count, replace=False)
    near_dups = df.iloc[near_src].copy()
    near_dups["country"] = rng.choice(["US", "UK", "IN", "DE", "FR"], near_dup_count).tolist()
    df = pd.concat([df, near_dups], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    gold = {
        "null_customer_id": real_null_cid,
        "null_email_real": real_null_email,
        "null_email_disguised": disguised_null_email,
        "null_email_total": real_null_email + disguised_null_email,
        "exact_duplicate_rows": dup_count,
        "near_duplicate_rows": near_dup_count,
    }
    return {"customers": df}, gold


def _task2(rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], dict]:
    n = 300
    amounts_float = (rng.random(n) * 500 + 5).round(2)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")[:n]
    df = pd.DataFrame(
        {
            "order_id": range(5001, 5001 + n),
            "customer_id": rng.integers(1001, 1201, n).tolist(),
            "amount": [f"${a}" for a in amounts_float],
            "order_date": [d.strftime("%b %d %Y") for d in dates],
            "status": rng.choice(["pending", "shipped", "delivered", "cancelled"], n).tolist(),
            "quantity": rng.integers(1, 20, n).tolist(),
        }
    )
    neg_qty = int(rng.integers(5, 11))
    neg_idx = rng.choice(n, neg_qty, replace=False)
    df.loc[neg_idx, "quantity"] = rng.integers(-10, 0, neg_qty).tolist()

    bad_amt = int(rng.integers(3, 8))
    bad_idx = rng.choice([i for i in range(n) if i not in set(neg_idx.tolist())], bad_amt, replace=False)
    df.loc[bad_idx, "amount"] = rng.choice(["N/A", "#ERR", "TBD", "--"], bad_amt).tolist()

    gold = {
        "amount_type_violation": True,
        "date_format_violation": True,
        "negative_quantity_rows": neg_qty,
        "unparseable_amount_rows": bad_amt,
    }
    return {"orders": df}, gold


def _task3(rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], dict]:
    def make_txn(n: int, rg: np.random.Generator, mean_amt: float, cats: list[str], id_start: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "txn_id": range(id_start, id_start + n),
                "user_id": rg.integers(2001, 2501, n).tolist(),
                "amount": rg.normal(mean_amt, 15, n).round(2).tolist(),
                "category": rg.choice(cats, n).tolist(),
                "ts": pd.date_range("2024-01-01", periods=n, freq="h")[:n].astype(str).tolist(),
            }
        )

    base_cats = ["food", "travel", "retail", "health", "utilities"]
    new_cats = ["crypto", "NFT"]

    baseline = make_txn(500, rng, mean_amt=50.0, cats=base_cats, id_start=10001)
    current_rng = np.random.default_rng(int(rng.integers(9999)))
    current = make_txn(500, current_rng, mean_amt=78.0, cats=base_cats + new_cats, id_start=10501)

    new_uid_count = int(0.15 * 500)
    new_uid_idx = current_rng.choice(500, new_uid_count, replace=False)
    current.loc[new_uid_idx, "user_id"] = current_rng.integers(3000, 3500, new_uid_count).tolist()

    gold = {
        "amount_mean_shift": True,
        "baseline_mean": 50.0,
        "current_mean": float(current["amount"].mean()),
        "new_categories": new_cats,
        "referential_drift_pct": new_uid_count / 500,
    }
    return {"transactions_baseline": baseline, "transactions_current": current}, gold


def _task4(rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], dict]:
    nc = 200
    customers = pd.DataFrame(
        {
            "customer_id": range(1, nc + 1),
            "name": [f"Customer {i}" for i in range(nc)],
            "tier": rng.choice(["bronze", "silver", "gold"], nc).tolist(),
        }
    )

    no = 500
    orphan_count = int(rng.integers(15, 22))
    valid_cids = list(range(1, nc + 1))
    order_cids = rng.choice(valid_cids, no - orphan_count).tolist()
    orphan_cids = rng.integers(9000, 9999, orphan_count).tolist()
    all_cids = order_cids + orphan_cids
    rng.shuffle(all_cids)

    order_dates = pd.date_range("2024-01-01", periods=no, freq="h")[:no]
    ship_dates = [d + pd.Timedelta(days=int(rng.integers(1, 10))) for d in order_dates]

    temp_viol = int(rng.integers(10, 16))
    temp_idx = rng.choice(no, temp_viol, replace=False)
    for i in temp_idx:
        ship_dates[i] = order_dates[i] - pd.Timedelta(days=int(rng.integers(1, 5)))

    orders = pd.DataFrame(
        {
            "order_id": range(1, no + 1),
            "customer_id": all_cids,
            "order_date": order_dates.astype(str).tolist(),
            "ship_date": [str(d) for d in ship_dates],
            "order_total": (rng.random(no) * 400 + 20).round(2).tolist(),
        }
    )

    nl = 1500
    li_order_ids = rng.choice(range(1, no + 1), nl).tolist()
    li_prices = (rng.random(nl) * 100 + 5).round(2)
    li_qtys = rng.integers(1, 6, nl)
    line_items = pd.DataFrame(
        {
            "line_id": range(1, nl + 1),
            "order_id": li_order_ids,
            "product": rng.choice(["Widget A", "Widget B", "Widget C", "Widget D"], nl).tolist(),
            "price": li_prices.tolist(),
            "quantity": li_qtys.tolist(),
            "subtotal": (li_prices * li_qtys).round(2).tolist(),
        }
    )

    agg_mismatch = int(rng.integers(5, 9))
    mismatch_order_ids = rng.choice(range(1, no + 1), agg_mismatch, replace=False)
    for oid in mismatch_order_ids:
        idx = orders[orders["order_id"] == oid].index
        if len(idx):
            orders.loc[idx[0], "order_total"] = round(float(orders.loc[idx[0], "order_total"]) * rng.uniform(1.3, 2.0), 2)

    gold = {
        "orphaned_order_count": orphan_count,
        "temporal_violation_count": temp_viol,
        "aggregate_mismatch_count": agg_mismatch,
        "total_orders": no,
    }
    return {"customers": customers, "orders": orders, "line_items": line_items}, gold
