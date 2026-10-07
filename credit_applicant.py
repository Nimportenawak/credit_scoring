from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --- Business Enums ---
class EmploymentType(Enum):
    FORMAL = "formal"  # Registered employee with contract
    INFORMAL = "informal"  # Gig economy/unregistered (Common in VN)
    SELF_EMPLOYED = "self_employed"  # Small business owner / Merchant
    UNEMPLOYED = "unemployed"


class LoanPurpose(Enum):
    CONSUMER = "consumer"  # Personal goods (e.g., motorbike, phone)
    BUSINESS = "business"  # Micro-enterprise funding (Common in HCMC)
    EDUCATION = "education"
    EMERGENCY = "emergency"  # Potential risk signal (urgent cash need)


# --- Main Structure ---
@dataclass
class CreditApplicant:
    # ── IDENTIFIER ────────────────────────────────────────
    applicant_id: str

    # ── BLOCK 1: DEMOGRAPHICS ─────────────────────────────
    age: int  # Range: [18, 70]
    province_code: str  # "HCM", "HAN", "DAN"... (Urban/Rural proxy)
    nb_dependents: int  # Number of dependents

    # ── BLOCK 2: FINANCIALS ───────────────────────────────
    monthly_income_vnd: float  # In VND
    monthly_expenses_vnd: float
    existing_debt_vnd: float  # Total outstanding debt
    loan_amount_requested_vnd: float
    loan_duration_months: int

    # Calculated Features (Derived during processing, not raw input)
    # → debt_to_income_ratio = (existing_debt + new_loan_installment) / monthly_income
    # → loan_to_income_ratio = loan_amount / monthly_income

    # ── BLOCK 3: EMPLOYMENT ───────────────────────────────
    employment_type: EmploymentType
    tenure_months: int  # Time at current job/position
    has_social_insurance: bool  # "BHXH" in VN — strong signal of formal work
    nb_income_sources: int  # Income diversification

    # ── BLOCK 4: BEHAVIORAL (Super-app Data) ──────────────
    app_tenure_days: int  # How long they've used the app
    avg_monthly_transactions: float  # Average transaction volume per month
    avg_transaction_value_vnd: float
    bill_payment_rate: float  # % of bills paid on time [0.0, 1.0]
    wallet_top_up_frequency: float  # Monthly top-ups
    has_linked_bank_account: bool  # Strong trust signal
    late_repayment_count: int  # Internal app credit history

    # ── BLOCK 5: LOAN CONTEXT ─────────────────────────────
    loan_purpose: LoanPurpose

    # ── TARGET (The variable to predict) ──────────────────
    # 1 = Default within 12 months, 0 = Good borrower
    label: Optional[int] = field(default=None)
