import numpy as np
from scipy.optimize import linprog
from ml_model import ROUTES

VESSELS = [
    {"type": "Feeder", "cap": 2500, "avail": 14, "cost_mult": 1.15},
    {"type": "Panamax", "cap": 5000, "avail": 12, "cost_mult": 1.00},
    {"type": "Post-Panamax", "cap": 10000, "avail": 8, "cost_mult": 0.88},
]


def optimize_capacity(forecast_rows):
    c = [r["cost_per_teu"] * v["cap"] * v["cost_mult"] for r in ROUTES for v in VESSELS]
    A_ub, b_ub = [], []

    for r_idx, f_row in enumerate(forecast_rows):
        row = [0.0] * 15
        for v_idx, v in enumerate(VESSELS):
            row[r_idx * 3 + v_idx] = -v["cap"]
        A_ub.append(row)
        b_ub.append(-f_row["forecast_teu"])

    for v_idx, v in enumerate(VESSELS):
        row = [0.0] * 15
        for r_idx in range(len(ROUTES)):
            row[r_idx * 3 + v_idx] = 1.0
        A_ub.append(row)
        b_ub.append(v["avail"])

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 15, integrality=np.ones(15), method="highs")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    allocations = []
    total_cap, total_demand = 0, 0
    for r_idx, f_row in enumerate(forecast_rows):
        counts = [int(round(res.x[r_idx * 3 + v_idx])) for v_idx in range(3)]
        cap_r = sum(counts[v_idx] * VESSELS[v_idx]["cap"] for v_idx in range(3))
        cost_r = sum(counts[v_idx] * c[r_idx * 3 + v_idx] for v_idx in range(3))
        total_cap += cap_r
        total_demand += f_row["forecast_teu"]
        allocations.append({
            "route_id": f_row["route_id"],
            "route_name": f_row["route_name"],
            "forecast_teu": f_row["forecast_teu"],
            "vessels": f"{counts[0]}F / {counts[1]}P / {counts[2]}PP",
            "allocated_cap_teu": cap_r,
            "utilization_pct": round(100.0 * f_row["forecast_teu"] / cap_r, 1),
            "est_cost_usd": int(round(cost_r)),
        })

    return {
        "allocations": allocations,
        "total_demand_teu": total_demand,
        "total_allocated_teu": total_cap,
        "overall_utilization_pct": round(100.0 * total_demand / total_cap, 1),
        "total_cost_usd": int(round(res.fun)),
    }