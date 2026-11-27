import numpy as np
import pandas as pd

ROUTES = [
    {"id": "R1", "name": "Asia-Europe", "base_teu": 18500, "growth": 180, "season_amp": 2200, "cost_per_teu": 420},
    {"id": "R2", "name": "Trans-Pacific", "base_teu": 21000, "growth": 210, "season_amp": 2800, "cost_per_teu": 460},
    {"id": "R3", "name": "Intra-Asia", "base_teu": 14500, "growth": 150, "season_amp": 1500, "cost_per_teu": 290},
    {"id": "R4", "name": "Middle East-India", "base_teu": 11200, "growth": 135, "season_amp": 1100, "cost_per_teu": 260},
    {"id": "R5", "name": "Trans-Atlantic", "base_teu": 12800, "growth": 110, "season_amp": 1400, "cost_per_teu": 380},
]

FEATURE_COLS = [
    "month_idx", "route_code", "sin_month", "cos_month",
    "fuel_price_idx", "port_congestion", "trade_volume_idx", "lag_demand_teu"
]


def load_or_generate_data(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    for r_idx, r in enumerate(ROUTES):
        prev_teu = r["base_teu"]
        for m in range(1, 25):
            moy = ((m - 1) % 12) + 1
            seasonal = r["season_amp"] * np.sin(2 * np.pi * (moy - 3) / 12.0)
            fuel_idx = 100 + 12 * np.sin(2 * np.pi * m / 18.0) + np.random.normal(0, 2.5)
            congestion_idx = float(np.clip(
                0.25 + 0.12 * np.cos(2 * np.pi * moy / 12.0) + np.random.normal(0, 0.03),
                0.10, 0.65
            ))
            trade_idx = 100 + 0.8 * m + np.random.normal(0, 1.8)
            demand = max(5000, round(
                r["base_teu"] + r["growth"] * m + seasonal
                + 45 * (trade_idx - 100) - 28 * (fuel_idx - 100)
                - 1800 * (congestion_idx - 0.25) + np.random.normal(0, 320), 0
            ))
            rows.append({
                "month_idx": m, "month": moy, "route_id": r["id"], "route_name": r["name"],
                "route_code": r_idx, "fuel_price_idx": round(fuel_idx, 2),
                "port_congestion": round(congestion_idx, 3), "trade_volume_idx": round(trade_idx, 2),
                "lag_demand_teu": prev_teu, "demand_teu": demand,
            })
            prev_teu = demand
    df = pd.DataFrame(rows)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


class DecisionTree:
    def __init__(self, max_depth=7, min_samples_split=3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _fit(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.std(y) < 1e-6:
            return float(np.mean(y))
        best_f, best_t, best_loss = None, None, float("inf")
        feat_idxs = np.random.choice(X.shape[1], max(1, int(np.sqrt(X.shape[1])) + 1), replace=False)
        for f in feat_idxs:
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 12:
                thresholds = np.quantile(thresholds, np.linspace(0.1, 0.9, 12))
            for t in thresholds:
                left, right = y[X[:, f] <= t], y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                loss = len(left) * np.var(left) + len(right) * np.var(right)
                if loss < best_loss:
                    best_loss, best_f, best_t = loss, int(f), float(t)
        if best_f is None:
            return float(np.mean(y))
        mask = X[:, best_f] <= best_t
        return {
            "feat": best_f, "thresh": best_t,
            "left": self._fit(X[mask], y[mask], depth + 1),
            "right": self._fit(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._fit(X, y, 0)

    def _pred_one(self, x, node):
        if not isinstance(node, dict):
            return node
        return self._pred_one(x, node["left"] if x[node["feat"]] <= node["thresh"] else node["right"])

    def predict(self, X):
        return np.array([self._pred_one(row, self.tree) for row in X])


class RandomForestRegressor:
    def __init__(self, n_estimators=80, max_depth=7, seed=42):
        self.n_estimators, self.max_depth, self.seed = n_estimators, max_depth, seed
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X), len(X), replace=True)
            t = DecisionTree(max_depth=self.max_depth)
            t.fit(X[idx], y[idx])
            self.trees.append(t)

    def predict(self, X):
        return np.mean(np.array([t.predict(X) for t in self.trees]), axis=0)


def train_models():
    df = load_or_generate_data()
    X, y = df[FEATURE_COLS].values, df["demand_teu"].values
    beta, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
    rf = RandomForestRegressor(n_estimators=80, max_depth=7, seed=42)
    rf.fit(X, y)
    return df, beta, rf


def forecast_horizon(df, beta, rf, target_month: int, fuel_idx: float, congestion_idx: float):
    lag_map = {r["id"]: float(df[df["route_id"] == r["id"]].iloc[-1]["demand_teu"]) for r in ROUTES}
    forecast_rows = []
    for m_idx in range(25, target_month + 1):
        moy = ((m_idx - 1) % 12) + 1
        trade_idx = 100.0 + 0.8 * m_idx
        current = []
        for r_idx, r in enumerate(ROUTES):
            x_next = np.array([[
                m_idx, r_idx, np.sin(2 * np.pi * moy / 12.0), np.cos(2 * np.pi * moy / 12.0),
                fuel_idx, congestion_idx, trade_idx, lag_map[r["id"]]
            ]])
            p_lr = float((np.column_stack([np.ones(1), x_next]) @ beta)[0])
            p_rf = float(rf.predict(x_next)[0])
            pred = int(round(0.55 * p_lr + 0.45 * p_rf))
            lag_map[r["id"]] = float(pred)
            current.append({"route_id": r["id"], "route_name": r["name"], "forecast_teu": pred})
        forecast_rows = current
    return forecast_rows