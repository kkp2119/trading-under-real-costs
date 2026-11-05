#!/usr/bin/env python3
"""Research 13 cost scenarios (SPEC 4, 3.5; E13-08: one position path, three cost ledgers).
  base   (C-base, verdict): sell at best_bid, buy at best_offer, $1.00 per contract per side, hedge 1 bp, kappa 1.25
  stress (C-stress):        mid -/+ 1.5 x half spread, $2.00, hedge 3 bp, kappa 1.50
  opt    (C-opt, reference): mid, $0, hedge 0.25 bp, kappa 1.00"""
import config13 as C


def sell_price(bid, offer, scen):
    mid, hs = (bid + offer) / 2.0, (offer - bid) / 2.0
    if scen == "base":
        return bid
    if scen == "stress":
        return mid - C.STRESS_HS * hs
    return mid


def buy_price(bid, offer, scen):
    mid, hs = (bid + offer) / 2.0, (offer - bid) / 2.0
    if scen == "base":
        return offer
    if scen == "stress":
        return mid + C.STRESS_HS * hs
    return mid


def fallback_buy(model_mid, scen):
    return C.KAPPA[scen] * model_mid


def fee(q, scen):
    return C.FEE[scen] * q


def hedge_cost(dH, S, scen):
    return C.HEDGE_BP[scen] * abs(dH) * S
