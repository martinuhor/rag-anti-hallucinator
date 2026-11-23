# ApexPay Global: Merchant Settlement & Compliance Operating Policy (v4.2)
**Document Classification:** Confidential & Regulatory Operations  
**Effective Date:** January 15, 2025 | **Review Frequency:** Quarterly  

---

## 1. Merchant Settlement Schedules & Reserve Rules

### 1.1 Settlement Cadence
- **Standard Settlement (T+2):** Merchant payout batches are finalized two business days post-transaction settlement date at 17:00 UTC with no surcharge.
- **Express Same-Day Settlement (T+0):** Available for approved merchants with over 6 months of processing history. Transactions processed before 11:00 UTC settle on the same calendar day for a flat 1.2% surcharge on the gross batch volume.
- **Weekend & Holiday Processing:** Standard settlements do not execute on Federal Reserve or TARGET2 bank holidays; funds roll over to the subsequent business day at 09:00 UTC.

### 1.2 Daily Volume Ceilings & Reserves
- **Tier 1 Merchant Cap:** Maximum daily aggregate settlement volume is capped at $250,000 USD equivalent.
- **Tier 2 Merchant Cap:** Maximum daily aggregate settlement volume is capped at $50,000 USD equivalent.
- **High-Risk Rolling Reserve:** Merchants assigned high-risk Merchant Category Codes (MCC 7995 for gaming/betting, MCC 5967 for direct marketing) are subject to a mandatory 5% rolling reserve withheld from each batch payout. Reserve funds are held in escrow for exactly 90 calendar days before automatic release to the merchant's operating account.

---

## 2. Chargeback Management, Dispute Windows & Fees

### 2.1 Dispute Timelines & Evidence Requirements
- **Pre-Arbitration Window:** Merchants have exactly 30 calendar days from the initial dispute notification date to accept liability or submit rebuttal documentation.
- **Evidence Submission Deadline:** Compelling evidence (signed delivery receipts, IP audit trails, CVV2 matching logs) must be uploaded via the Merchant Portal within 14 calendar days of dispute initiation. Submissions past 14 days are auto-forfeited.
- **Dispute Processing Fee:** A non-refundable fee of $15.00 USD is assessed per chargeback case, regardless of whether the dispute is won, lost, or withdrawn.

### 2.2 Chargeback Ratio Enforcement
- **Standard Threshold:** The merchant chargeback ratio is calculated as `(Total Monthly Dispute Count / Total Monthly Transaction Count) * 100`.
- **Surveillance Trigger:** A ratio exceeding 0.90% in any rolling 30-day window places the merchant on Probationary Surveillance Status for 60 calendar days.
- **Breach Penalty:** If the chargeback ratio remains above 1.20% for two consecutive months, ApexPay enforces an immediate 10% liquidity holdback and suspends card-not-present (CNP) routing.

---

## 3. Know Your Customer (KYC) & Tier Verification Limits

### 3.1 Verification Tiers
- **Tier 1 (Basic Account):** Verified via authenticated SMS OTP and business email. Maximum lifetime transacting volume is $1,000 USD. Outbound cross-border withdrawals are disabled.
- **Tier 2 (Standard Verified):** Requires a government-issued photo ID (Passport or Driver's License) and a certified utility bill or bank statement dated within the preceding 90 calendar days. Grants a processing limit of up to $25,000 USD per calendar month.
- **Tier 3 (Enterprise / High Volume):** Requires full Enhanced Due Diligence (EDD), Corporate Articles of Incorporation, and Ultimate Beneficial Owner (UBO) declaration for all entities holding 25% or greater equity. Tier 3 unlocks unlimited transaction volume, custom settlement rails, and requires an annual AML audit recertification.

---

## 4. Cross-Border Payment Rails & API Error Codes

### 4.1 System Error Codes & Handling Procedures
- **`ERR_FX_SLIP_409` (Foreign Exchange Slippage):** Triggered when currency exchange rate market drift exceeds 0.35% between the initial quote generation and transaction execution. The API caller must re-quote and execute within a 60-second validity window.
- **`ERR_SANCTION_BLOCK_403` (Sanctions Compliance Match):** Triggered when the sender or beneficiary account matches entries on OFAC, EU, or UN sanctions watchlists. The payment rail automatically freezes the escrowed funds and alerts the Compliance Officer within 4 hours.
- **`ERR_SETTLE_CAP_422` (Corridor Ceiling Exceeded):** Triggered when an individual batch transfer exceeds the designated corridor settlement ceiling: $100,000 USD for SEPA transfers or $50,000 USD for cross-border ACH wires.
- **`ERR_VELOCITY_BREACH_429` (Rate Limit Exceeded):** Occurs when an API key submits more than 500 requests per minute. Merchants must implement client-side exponential backoff with randomized jitter to resume requests.
