"""Deterministic synthetic dataset for a consumer-lending feature platform.

Design goals
------------
* Two years of users / applications / loans / repayments with realistic
  seasonality, growth and risk mix drift.
* Explicitly planted temporal traps (see TRAPS below), each of which makes a
  naive feature implementation disagree with a point-in-time one.
* Every row carries ``ingested_at`` so the audit can reconstruct exactly what
  the warehouse knew at any wall-clock instant, not just what the event time
  says.  That distinction is what separates "event time" from "knowledge time"
  and it is where most real leaks hide.

TRAPS
-----
T1  apply_time_local   naive parsing shifts event time by +8h.
T2  decision_time      application status / reject_reason only exist after the
                       decision; using them for young applications leaks.
T3  loans duplicate    append-only ingest log, ~2% of loans restated later with
                       a different principal.  Dedupe on (loan_id, max ingested_at).
T4  cancelled loans    final loan_status='CANCELLED' is future knowledge at
                       early as-of times.
T5  reversed payments  final repayment status='REVERSED' is future knowledge.
T6  profile history    SCD2; latest snapshot must not be used as-of an earlier time.
T7  feed backfill      partner-channel applications/loans arrive 1-45 days late,
                       so ingestion-date partitions != event date.
T8  label readiness    labels only exist once the 90-day observation window closes.
"""
from __future__ import annotations

import argparse
import math


import duckdb
import numpy as np
import pandas as pd

from feature_platform import config

EPS = 1e-9

CHANNELS = ["app_store", "ad_network", "partner_bank", "offline_store", "referral"]
CHANNEL_W = [0.34, 0.26, 0.19, 0.11, 0.10]
CITIES = ["tier1", "tier2", "tier3", "tier4"]
CITY_W = [0.28, 0.34, 0.25, 0.13]
OCCUPATIONS = ["salaried", "self_employed", "freelance", "student", "retired"]
OCC_W = [0.46, 0.22, 0.18, 0.09, 0.05]
EDUCATION = ["high_school", "college", "bachelor", "master_plus"]
EDU_W = [0.22, 0.31, 0.37, 0.10]
DEVICES = ["android_low", "android_mid", "android_high", "ios"]
DEVICE_W = [0.18, 0.37, 0.27, 0.18]
PRODUCTS = ["cash_loan", "installment", "credit_line"]
PRODUCT_W = [0.55, 0.31, 0.14]
APP_VERSIONS = ["5.1.0", "5.2.0", "5.3.1", "6.0.0", "6.1.2"]


def _ts(s: str) -> np.datetime64:
    return np.datetime64(s, "s")


def _ts_str(t: np.datetime64) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _poisson(rng, lam):
    return rng.poisson(max(lam, 0.0))


def generate(seed: int = config.SEED, verbose: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    today = _ts(config.TODAY) + np.timedelta64(23, "h")
    user_start = _ts("2023-03-01")

    # ---------------------------------------------------------------- users
    n_users = 24_000
    reg_offsets = rng.integers(0, int((_ts("2025-10-31") - user_start) / np.timedelta64(1, "D")), n_users)
    _risk = rng.normal(0.0, 1.0, n_users)
    # 2025 cohort drifts riskier and more app-heavy (produces distribution drift)
    reg_days = reg_offsets.astype(np.int64)
    drift_boost = np.where(reg_days > 700, 0.35, 0.0)
    _risk = _risk + drift_boost + rng.normal(0, 0.15, n_users)
    _income = np.clip(rng.lognormal(10.2, 0.45, n_users), 25_000, 900_000)
    _credit = np.clip(720 - 78 * _risk + rng.normal(0, 32, n_users), 480, 850).astype(int)
    # payment discipline is a separate, persistent user trait that both drives
    # past repayment behaviour and predicts future delinquency -- this is what
    # makes repayment-history features genuinely predictive (as in real data).
    _discipline = 0.55 * _risk + rng.normal(0, 0.85, n_users)

    users = pd.DataFrame({
        "user_id": [f"U{i:07d}" for i in range(n_users)],
        "register_time": [_ts_str(user_start + np.timedelta64(int(d), "D") + np.timedelta64(int(rng.integers(0, 86400)), "s")) for d in reg_days],
        "channel": rng.choice(CHANNELS, n_users, p=CHANNEL_W),
        "city_tier": rng.choice(CITIES, n_users, p=CITY_W),
        "age": np.clip(rng.normal(33, 8.5, n_users), 18, 70).astype(int),
        "gender": rng.choice(["F", "M", "unknown"], n_users, p=[0.43, 0.54, 0.03]),
        "occupation": rng.choice(OCCUPATIONS, n_users, p=OCC_W),
        "education": rng.choice(EDUCATION, n_users, p=EDU_W),
        "income_level": np.round(_income).astype(int),
        "device_type": rng.choice(DEVICES, n_users, p=DEVICE_W),
        "is_employee": rng.random(n_users) < 0.71,
    })
    users["register_time"] = pd.to_datetime(users["register_time"])

    # ------------------------------------------------------ profile history
    prof_rows = []
    snap_offsets_days = [0, 90, 180, 270, 365, 450, 540, 630, 720, 810, 900, 990]
    for i in range(n_users):
        uid = users.at[i, "user_id"]
        reg = users.at[i, "register_time"]
        base_credit = int(_credit[i])
        risk = float(_risk[i])
        blacklist_until = None
        # occasional blacklist episode, recorded at the snapshot that covers it
        if rng.random() < 0.012:
            start_off = int(rng.integers(30, 700))
            blacklist_until = reg + pd.Timedelta(days=start_off + int(rng.integers(60, 240)))
        credit = base_credit
        dev = users.at[i, "device_type"]
        ver = str(rng.choice(APP_VERSIONS))
        income = int(users.at[i, "income_level"])
        for k, off in enumerate(snap_offsets_days):
            st = reg + pd.Timedelta(days=off)
            if st > pd.Timestamp(today):
                break
            # quarterly profile refresh: score walk, occasional device/version change
            if k > 0:
                credit = int(np.clip(credit + rng.normal(-6 + 4 * risk, 28), 440, 880))
                if rng.random() < 0.12:
                    dev = str(rng.choice(DEVICES))
                if rng.random() < 0.35:
                    ver = str(rng.choice(APP_VERSIONS))
                if rng.random() < 0.10:
                    income = int(np.clip(income * rng.uniform(0.9, 1.18), 20_000, 1_200_000))
            is_bl = blacklist_until is not None and st <= blacklist_until < st + pd.Timedelta(days=90)
            risk_level = "high" if credit < 580 else ("medium" if credit < 680 else "low")
            # ingest delay for this snapshot feed
            ingested = st + pd.Timedelta(hours=int(rng.integers(1, 36)))
            prof_rows.append((
                f"P{i:07d}_{k}", uid, _ts_str(np.datetime64(st)), credit, risk_level,
                income, dev, ver, bool(is_bl), _ts_str(np.datetime64(ingested)),
            ))
    profile_hist = pd.DataFrame(prof_rows, columns=[
        "profile_id", "user_id", "snapshot_time", "credit_score", "risk_level",
        "income_level", "device_type", "app_version", "is_blacklist", "ingested_at",
    ])
    profile_hist["snapshot_time"] = pd.to_datetime(profile_hist["snapshot_time"])
    profile_hist["ingested_at"] = pd.to_datetime(profile_hist["ingested_at"])

    # ----------------------------------------------------------- applications
    app_rows = []
    loan_rows = []          # logical loans, before ingest duplication
    repay_rows = []
    label_rows = []

    user_ids = users["user_id"].to_numpy()
    reg_times = users["register_time"].to_numpy()
    user_channel = users["channel"].to_numpy()
    user_dev = users["device_type"].to_numpy()
    user_income = users["income_level"].to_numpy()

    loan_seq = 0
    app_seq = 0
    for i in range(n_users):
        reg = pd.Timestamp(reg_times[i])
        active_days = max((pd.Timestamp(today) - reg).days, 30)
        lam = 1.6 + 1.4 * (1 / (1 + math.exp(-_risk[i])))
        n_apps = int(np.clip(rng.poisson(lam), 0, 14))
        if n_apps == 0:
            continue
        # activity spread across lifetime, with recency bias
        gaps = rng.exponential(active_days / max(n_apps, 1), n_apps)
        gaps = np.sort(gaps)
        for g in gaps:
            at = reg + pd.Timedelta(days=float(g))
            if at > pd.Timestamp(today - pd.Timedelta(days=1)):
                continue
            app_seq += 1
            aid = f"A{app_seq:08d}"
            uid = user_ids[i]
            # --- risk mix drifts upward in 2025 (drift signal for monitoring)
            year_boost = 0.45 if at >= pd.Timestamp("2025-06-01") else 0.0
            latent = float(_risk[i]) + year_boost
            product = str(rng.choice(PRODUCTS, p=PRODUCT_W))
            base_amt = float(_income[i]) * rng.uniform(0.05, 0.35)
            if product == "credit_line":
                base_amt *= 0.55
            amt = float(np.clip(round(base_amt / 100) * 100, 500, 120_000))
            dev = str(user_dev[i])
            if rng.random() < 0.06:
                dev = str(rng.choice(DEVICES))
            ver = str(rng.choice(APP_VERSIONS))
            if rng.random() < 0.08:
                ver = "unknown"
            # ---------------------------- decision: manual review subset (T2)
            auto = rng.random() > 0.15
            if auto:
                decision_delay = pd.Timedelta(minutes=int(rng.integers(1, 45)))
            else:
                decision_delay = pd.Timedelta(hours=int(rng.integers(3, 73)))
            p_approve = 1 / (1 + math.exp(-(1.25 - 0.85 * latent + 0.6 * (user_income[i] / 300_000 - 1))))
            approved = bool(rng.random() < p_approve)
            status = "APPROVED" if approved else "REJECTED"
            reason = None if approved else str(rng.choice([
                "risk_score_low", "insufficient_income", "fraud_suspect", "over_indebted",
            ], p=[0.42, 0.27, 0.13, 0.18]))
            decision_time = at + decision_delay
            # Feed delay (T7).  Direct app channels are recorded essentially in
            # real time (a few seconds).  Partner-bank / offline channels are
            # batch backfills that arrive 1-45 days late; that latency is the
            # realistic source of "the warehouse did not know yet at as_of".
            if user_channel[i] in ("partner_bank", "offline_store"):
                feed_delay = pd.Timedelta(hours=float(rng.integers(24, 1080)))
            else:
                feed_delay = pd.Timedelta(seconds=float(rng.uniform(0, 4)))
            ingested_at = at + feed_delay
            local = at.tz_localize("UTC").tz_convert("Asia/Shanghai").isoformat()
            ip_city = None if rng.random() < 0.015 + (0.08 if at >= pd.Timestamp("2025-05-01") else 0.0) else str(rng.choice(CITIES, p=CITY_W))
            app_rows.append((aid, uid, _ts_str(np.datetime64(at)), local, str(user_channel[i]),
                             dev, ver, product, amt, status, reason,
                             _ts_str(np.datetime64(decision_time)), ip_city, _ts_str(np.datetime64(ingested_at))))

            if not approved:
                continue
            loan_seq += 1
            lid = f"L{loan_seq:08d}"
            # disburse shortly after decision
            loan_time = decision_time + pd.Timedelta(minutes=int(rng.integers(2, 240)))
            if loan_time > pd.Timestamp(today):
                continue
            principal = round(amt * rng.uniform(0.72, 1.0), 2)
            term = int(rng.choice([90, 180, 360], p=[0.34, 0.43, 0.23]))
            rate = round(float(np.clip(rng.normal(0.18 + 0.06 * latent, 0.035), 0.06, 0.36)), 4)
            cancelled = rng.random() < 0.035
            cancelled_time = loan_time + pd.Timedelta(hours=float(rng.uniform(1, 72))) if cancelled else None
            if cancelled_time is not None and cancelled_time > pd.Timestamp(today):
                cancelled_time = None
                cancelled = False
            loan_status = "CANCELLED" if cancelled else "ACTIVE"
            # outcome: latent risk + amount/income ratio + product effects
            util = amt / max(_income[i], 1)
            # observed risk drivers: bureau score (via latent), utilisation,
            # prior repayment discipline, and borrowing intensity so far
            prior_loans = sum(1 for r in loan_rows if r[2] == uid)
            prior_apps = sum(1 for r in app_rows if r[1] == uid)
            dpd_latent = (
                -2.35
                + 0.85 * latent
                + 1.25 * util
                + 0.95 * float(_discipline[i])
                + 0.075 * min(prior_loans, 8)
                + 0.045 * min(prior_apps, 10)
                + 0.30 * (product == "credit_line")
                - 0.20 * (term / 360)
            )
            p_bad = 1 / (1 + math.exp(-dpd_latent))
            bad = rng.random() < p_bad
            if bad:
                u = rng.random()
                final_dpd = int(rng.choice([30, 45, 60, 90, 120], p=[0.35, 0.22, 0.2, 0.15, 0.08])) if u < 0.72 else int(rng.integers(1, 30))
            else:
                final_dpd = 0 if rng.random() < 0.93 else int(rng.integers(1, 10))
            if cancelled:
                final_dpd = 0
            # loan ingest: same feed delay, plus restatement duplication (T3)
            base_ingested = loan_time + feed_delay
            lender_principal = principal
            if rng.random() < 0.08:
                lender_principal = round(principal * rng.uniform(0.97, 1.03), 2)  # restatement
            loan_rows.append((lid, aid, uid, loan_time, lender_principal, term, rate, product,
                              loan_status, cancelled_time, final_dpd, base_ingested))
            if lender_principal != principal:
                # duplicate row arrives weeks later correcting the principal
                loan_rows.append((lid, aid, uid, loan_time, principal, term, rate, product,
                                  loan_status, cancelled_time, final_dpd,
                                  base_ingested + pd.Timedelta(days=int(rng.integers(7, 40)))))
            # ------------------------------ repayment schedule
            n_inst = {90: 3, 180: 6, 360: 12}[term]
            inst_gap = term // n_inst
            is_reversed_inst = int(rng.integers(0, n_inst)) if rng.random() < 0.018 else -1
            for k in range(n_inst):
                due = loan_time + pd.Timedelta(days=inst_gap * (k + 1))
                amt_due = round(lender_principal / n_inst * (1 + rate * inst_gap / 365), 2)
                rid = f"R{loan_seq:08d}{k:02d}"
                if cancelled:
                    pay_status = "CANCELLED"
                    repay_time = None
                    amount_paid = 0.0
                    dpd = None
                    reversal = None
                else:
                    # payment behaviour: bad loans go late as delinquency accumulates
                    disc = float(_discipline[i])
                    if bad and final_dpd >= 30:
                        late = int(rng.integers(25, 95)) if k >= 1 else int(rng.integers(0, 40))
                    elif final_dpd > 0:
                        late = int(rng.integers(0, 18))
                    else:
                        late = int(np.clip(rng.normal(-4 + 9 * disc, 6), -12, 35))
                    repay_time = due + pd.Timedelta(days=late)
                    if repay_time > pd.Timestamp(today):
                        repay_time = None
                    if repay_time is None:
                        pay_status = "OVERDUE"
                        amount_paid = 0.0
                        dpd = int((pd.Timestamp(today) - due).days)
                    else:
                        pay_status = "PAID"
                        amount_paid = amt_due if rng.random() > 0.06 else round(amt_due * rng.uniform(0.3, 0.95), 2)
                        dpd = max(late, 0)
                    reversal = None
                    if k == is_reversed_inst and repay_time is not None:
                        reversal = repay_time + pd.Timedelta(hours=float(rng.uniform(2, 96)))
                        if reversal > pd.Timestamp(today):
                            reversal = None
                ingested_r = (repay_time if repay_time is not None else min(due + pd.Timedelta(days=1), pd.Timestamp(today))) \
                    + pd.Timedelta(minutes=float(rng.integers(1, 240)))
                if rng.random() < 0.01:          # occasional late-arriving payment feed
                    ingested_r += pd.Timedelta(days=int(rng.integers(2, 20)))
                repay_rows.append((rid, lid, uid, k + 1, _ts_str(np.datetime64(due)),
                                   _ts_str(np.datetime64(repay_time)) if repay_time is not None else None,
                                   amt_due, amount_paid, dpd, pay_status,
                                   _ts_str(np.datetime64(reversal)) if reversal is not None else None,
                                   _ts_str(np.datetime64(loan_time)),
                                   _ts_str(np.datetime64(ingested_r))))
            # ------------------------------ labels (T8)
            obs_end = loan_time + pd.Timedelta(days=config.LABEL_WINDOW_DAYS)
            label_ready = obs_end + pd.Timedelta(days=7)  # T+7 settlement lag
            if cancelled:
                label_rows.append((aid, lid, None, None, None, _ts_str(np.datetime64(obs_end)), "cancelled"))
            elif label_ready <= pd.Timestamp(today):
                max_dpd_90 = int(max(0, min(final_dpd, config.LABEL_WINDOW_DAYS)))
                label = int(final_dpd >= config.LABEL_DPD_THRESHOLD)
                label_rows.append((aid, lid, label, max_dpd_90, _ts_str(np.datetime64(obs_end)),
                                   _ts_str(np.datetime64(label_ready)), "observed"))
            else:
                label_rows.append((aid, lid, None, None, _ts_str(np.datetime64(obs_end)), None, "immature"))

    if verbose:
        print(f"users={len(users)} apps={len(app_rows)} loans={len(loan_rows)} "
              f"repay={len(repay_rows)} labels={len(label_rows)}")

    apps = pd.DataFrame(app_rows, columns=[
        "application_id", "user_id", "apply_time", "apply_time_local", "channel",
        "device_type", "app_version", "product_id", "apply_amount", "apply_status",
        "reject_reason", "decision_time", "ip_city", "ingested_at",
    ])
    for c in ("apply_time", "decision_time", "ingested_at"):
        apps[c] = pd.to_datetime(apps[c])

    loans = pd.DataFrame(loan_rows, columns=[
        "loan_id", "application_id", "user_id", "loan_time", "principal", "term_days",
        "interest_rate", "product_id", "loan_status", "cancelled_time", "final_dpd", "ingested_at",
    ])
    loans["loan_time"] = pd.to_datetime(loans["loan_time"])
    loans["cancelled_time"] = pd.to_datetime(loans["cancelled_time"])
    loans["ingested_at"] = pd.to_datetime(loans["ingested_at"])

    commitments = pd.DataFrame(repay_rows, columns=[
        "repayment_id", "loan_id", "user_id", "installment_no", "due_time", "repay_time",
        "amount_due", "amount_paid", "dpd_days", "status", "reversal_time",
        "schedule_known_at", "ingested_at",
    ])
    for c in ("due_time", "repay_time", "reversal_time", "schedule_known_at", "ingested_at"):
        commitments[c] = pd.to_datetime(commitments[c])

    labels = pd.DataFrame(label_rows, columns=[
        "application_id", "loan_id", "label", "max_dpd_90d", "observation_end",
        "label_ready_time", "label_reason",
    ])
    for c in ("observation_end", "label_ready_time"):
        labels[c] = pd.to_datetime(labels[c])

    return {
        "users": users,
        "user_profile_history": profile_hist,
        "applications": apps,
        "loans": loans,
        "repayments": commitments,
        "labels": labels,
    }


def write_to_duckdb(tables: dict[str, pd.DataFrame], db_path=None) -> None:
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute(Path(__file__).with_name("schema.sql").read_text())
    for name, df in tables.items():
        con.register("_tmp", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM _tmp")
        con.unregister("_tmp")
    con.execute("CHECKPOINT")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    tables = generate(seed=args.seed, verbose=not args.quiet)
    write_to_duckdb(tables)
    print(f"wrote {config.DB_PATH} ({config.DB_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    from pathlib import Path
    main()
