# src/riskquant/basel/constants/irb.py
import math

# --- IRB Corporate, Sovereign, Bank Constants ---
# Ref: Basel Consolidated Framework (CRE31.9)
IRB_CORRELATION_LOWER_BOUND: float = 0.12
IRB_CORRELATION_UPPER_BOUND: float = 0.24
IRB_CORRELATION_DECAY_FACTOR: float = -50.0
IRB_CORRELATION_DENOMINATOR: float = 1.0 - math.exp(IRB_CORRELATION_DECAY_FACTOR)

# --- High-Volatility Commercial Real Estate (HVCRE) ---
# Ref: Basel Consolidated Framework (CRE31.11)
IRB_HVCRE_CORRELATION_UPPER_BOUND: float = 0.30

# --- Firm-Size Adjustment Constants (SME) ---
# Ref: Basel Consolidated Framework (CRE31.10)
IRB_SME_SALES_MIN_EUR_M: float = 5.0
IRB_SME_SALES_MAX_EUR_M: float = 50.0
IRB_SME_ADJ_DIVISOR: float = 45.0
IRB_SME_ADJ_MAX_MULTIPLIER: float = 0.04

# --- Retail: Other Retail Constants ---
# Ref: Basel Consolidated Framework (CRE31.14)
IRB_RETAIL_OTHER_LOWER_BOUND: float = 0.03
IRB_RETAIL_OTHER_UPPER_BOUND: float = 0.16
IRB_RETAIL_OTHER_DECAY_FACTOR: float = -35.0
IRB_RETAIL_OTHER_DENOMINATOR: float = 1.0 - math.exp(IRB_RETAIL_OTHER_DECAY_FACTOR)

# --- Retail: Fixed Scalar Constants ---
# Ref: Basel Consolidated Framework (CRE31.14)
IRB_RETAIL_MORTGAGE_CORRELATION: float = 0.15
IRB_RETAIL_QRRE_CORRELATION: float = 0.04
