"""Share and cash accounting isolated from the legacy equal-weight simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_loader import BacktestDataset
from providers.max_score import SnapshotDataError


@dataclass(slots=True)
class ShareAccount:
    shares: np.ndarray
    prices: np.ndarray
    bought_day: np.ndarray
    cash: float

    @classmethod
    def empty(cls, n_symbols: int, initial_cash: float) -> ShareAccount:
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("portfolio_initial_value must be positive and finite")
        return cls(np.zeros(n_symbols), np.full(n_symbols, np.nan),
                   np.full(n_symbols, -1, dtype=np.int64), float(initial_cash))

    @property
    def nav(self) -> float:
        held = self.shares > 0
        if np.any(~np.isfinite(self.prices[held])) or np.any(self.prices[held] <= 0):
            raise SnapshotDataError("MISSING_VALUATION", "held asset has no positive valuation price")
        nav = float(self.cash + np.dot(self.shares[held], self.prices[held]))
        if not np.isfinite(nav) or nav <= 0 or self.cash < -1e-7:
            raise SnapshotDataError("INVALID_ACCOUNT_NAV", "invalid NAV or negative cash")
        return nav

    def mark_close(self, pool: BacktestDataset, bar_idx: int, *, missing_policy: str) -> int:
        start, end = map(int, pool.bar_offsets[bar_idx:bar_idx + 2])
        rows = np.arange(start, end)
        valid = rows[np.isfinite(pool.bar_close[rows]) & (pool.bar_close[rows] > 0)]
        present = np.zeros(len(self.shares), dtype=bool)
        symbols = pool.row_symbol_ids[valid]
        present[symbols] = True
        missing = (self.shares > 0) & ~present
        if missing.any() and missing_policy == "error":
            raise SnapshotDataError("MISSING_VALUATION", f"{int(missing.sum())} held assets lack this close")
        self.prices[symbols] = pool.bar_close[valid]
        self.nav  # validate carried marks too
        return int(missing.sum())

    def weights(self, symbol_ids: np.ndarray) -> np.ndarray:
        result = np.zeros(len(symbol_ids), dtype=np.float64)
        held = self.shares[symbol_ids] > 0
        result[held] = self.shares[symbol_ids[held]] * self.prices[symbol_ids[held]] / self.nav
        return result


def execute_next_bar(account: ShareAccount, pool: BacktestDataset, bar_idx: int,
                     symbol_ids: np.ndarray, planned_shares: np.ndarray, *,
                     cost_rate: float, priority: np.ndarray) -> dict:
    """Execute prior-close quantities, sells first, without borrowing cash.

    Feasibility is binary, while available cash can partly fund an order. The
    caller owns the immutable desired quantities; a failed order never changes
    those targets or pretends to have filled them.
    """
    if not np.isfinite(cost_rate) or cost_rate < 0 or cost_rate >= 1:
        raise ValueError("cost_per_turnover must be finite and in [0, 1)")
    before = account.nav
    start, end = map(int, pool.bar_offsets[bar_idx:bar_idx + 2])
    row_map = {int(pool.row_symbol_ids[r]): r for r in range(start, end)}
    day = int(np.datetime64(pool.bars[bar_idx], "D").astype(np.int64))
    desired = dict(zip(map(int, symbol_ids), map(float, planned_shares)))
    trades: list[dict] = []
    notional, fees, closed = 0.0, 0.0, 0
    ordered_ids = list(map(int, symbol_ids[priority]))
    for side in ("sell", "buy"):
        for sid in ordered_ids:
            delta = desired[sid] - account.shares[sid]
            if (side == "sell" and delta >= 0) or (side == "buy" and delta <= 0):
                continue
            row = row_map.get(sid)
            reason = None
            if row is None:
                reason = "missing_execution_row"
            else:
                price = float(pool.execution_vwap[row])
                if not np.isfinite(price) or price <= 0:
                    reason = "invalid_execution_price"
                elif side == "sell" and (not pool.can_trade_sell[row] or account.bought_day[sid] >= day):
                    reason = "sell_blocked"
                elif side == "buy" and (not pool.can_trade_buy[row] or not pool.can_open_base[row]):
                    reason = "buy_blocked"
            if reason:
                trades.append({"symbol_id": sid, "side": side, "shares": 0.0, "notional": 0.0,
                               "fee": 0.0, "status": reason, "requested_shares": abs(delta)})
                continue
            quantity = min(-delta, account.shares[sid]) if side == "sell" else min(
                delta, max(0.0, account.cash) / (price * (1.0 + cost_rate)))
            amount, fee = quantity * price, quantity * price * cost_rate
            if side == "buy":
                # Division/multiplication can round an affordable quantity one
                # ulp above available cash. Reduce the executed quantity by
                # representable steps; do not clip cash or invent financing.
                while amount + fee > account.cash and quantity > 0:
                    quantity = float(np.nextafter(quantity, 0.0))
                    amount, fee = quantity * price, quantity * price * cost_rate
            if side == "sell":
                account.shares[sid] -= quantity
                account.cash += amount - fee
                if account.shares[sid] == 0:
                    closed += 1
            else:
                account.shares[sid] += quantity
                account.cash -= amount + fee
                if quantity > 0:
                    account.bought_day[sid] = day
                    # Needed only until the close mark; execution price is the
                    # first known mark for a newly purchased asset.
                    if not np.isfinite(account.prices[sid]):
                        account.prices[sid] = price
            notional += amount
            fees += fee
            trades.append({"symbol_id": sid, "side": side, "shares": quantity,
                           "notional": amount, "fee": fee,
                           "status": "filled" if quantity == abs(delta) else "partial_cash",
                           "requested_shares": abs(delta)})
    return {"notional": notional, "fees": fees, "turnover": notional / before,
            "closed": closed, "trades": trades}
