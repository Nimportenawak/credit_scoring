import math
import random as rd

from typing_extensions import Optional

from credit_applicant import CreditApplicant, EmploymentType, LoanPurpose


def marsaglia_tsang(a: float) -> float:
    if a < 1:
        return marsaglia_tsang(a + 1) * (rd.random() ** (1.0 / a))

    d = a - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        Z = rd.gauss(0, 1)
        V = (1.0 + c * Z) ** 3

        if V <= 0:
            continue
        U = rd.random()
        if U < (1.0 - 0.0331 * Z**4):
            return d * V

        if math.log(U) < 0.5 * (Z**2) + d * (1.0 - V + math.log(V)):
            return d * V


def sample_beta(alpha: float, beta: float) -> float:
    x = marsaglia_tsang(alpha)
    y = marsaglia_tsang(beta)
    return x / (x + y)


def sample_poisson(lam: float) -> int:
    U = rd.random()
    p = math.exp(-lam)
    F = p
    k = 0

    while U > F:
        k += 1
        p *= lam / k
        F += p
    return k


def sample_dependants(age: int) -> int:
    if age < 25:
        lam = 0.3
    elif age < 40:
        lam = 1.5
    else:
        lam = 2.0
    return sample_poisson(lam)


def sample_lognormal(mu: float, sigma_income: float) -> float:
    return math.exp(rd.gauss(mu, sigma_income))


def sigmoid(z: float) -> float:
    return 1 / 1 + math.exp(-z)


def generate_synthetic_dataset(n: int):
    dataset = []
    provinces_codes = [
        "HANOI",
        "HCM",
        "DANANG",
        "HAIPHONG",
        "CANTHO",
        "CAOBANG",
        "LANGSON",
        "QUANGNINH",
        "THAIBINH",
        "NAMDINH",
        "PHUTHO",
        "THAINGUYEN",
        "YENBAI",
        "TUYENQUANG",
        "HAGIANG",
        "LAOCAI",
        "LAICHAU",
        "SONLA",
        "DIENBIEN",
        "HOABINH",
        "HAIDUONG",
        "NINHBINH",
        "THANHHOA",
        "NGHEAN",
        "HATINH",
        "QUANGBINH",
        "QUANGTRI",
        "HUE",
        "QUANGNAM",
        "QUANGNGAI",
        "BINHDINH",
        "PHUYEN",
        "KHANHHOA",
        "NINHTHUAN",
        "BINHTHUAN",
        "KON TUM",
        "GIALAI",
        "DAKLAK",
        "DAKNONG",
        "LAMDONG",
        "BINHPHUOC",
        "TAYNINH",
        "BINHDUONG",
        "DONGNAI",
        "BARIAVUNGTAU",
        "LONGAN",
        "TIENGIANG",
        "BENTRE",
        "TRAVINH",
        "VINHLONG",
        "DONGTHAP",
        "ANGIANG",
        "KIENGIANG",
        "CANTHO",
        "HAUGIANG",
        "SOCTRANG",
        "BACLIEU",
        "CAMAU",
        "BACKAN",
        "BACGIANG",
        "BACNINH",
        "VINHPHUC",
        "HUNGYEN",
    ]

    INCOME_PARAMS = {
        EmploymentType.FORMAL: (12_000_000, 0.4),
        EmploymentType.INFORMAL: (6_000_000, 0.5),
        EmploymentType.SELF_EMPLOYED: (8_000_000, 0.6),
        EmploymentType.UNEMPLOYED: (1_500_000, 0.3),
    }

    RATIO_PARAMS = {
        EmploymentType.FORMAL: (3, 7),
        EmploymentType.INFORMAL: (5, 5),
        EmploymentType.SELF_EMPLOYED: (4, 5),
        EmploymentType.UNEMPLOYED: (7, 3),
    }

    DEBT_PARAMS = {
        #                        médiane du multiple  σ
        EmploymentType.FORMAL: (1.0, 0.5),
        EmploymentType.INFORMAL: (2.0, 0.6),
        EmploymentType.SELF_EMPLOYED: (2.5, 0.7),
        EmploymentType.UNEMPLOYED: (4.0, 0.5),
    }

    LOAN_PARAMS = {
        EmploymentType.FORMAL: (3.0, 0.5),
        EmploymentType.INFORMAL: (2.0, 0.6),
        EmploymentType.SELF_EMPLOYED: (2.5, 0.5),
        EmploymentType.UNEMPLOYED: (1.0, 0.4),
    }

    BILL_PARAMS = {
        EmploymentType.FORMAL: (8, 2),  # moyenne 0.80
        EmploymentType.INFORMAL: (4, 4),  # moyenne 0.50
        EmploymentType.SELF_EMPLOYED: (5, 4),  # moyenne 0.56
        EmploymentType.UNEMPLOYED: (2, 6),  # moyenne 0.25
    }

    INSURANCE_PARAMS = {
        EmploymentType.FORMAL: 0.85,
        EmploymentType.INFORMAL: 0.30,
        EmploymentType.SELF_EMPLOYED: 0.10,
        EmploymentType.UNEMPLOYED: 0.05,
    }

    TOPUP_PARAMS = {
        EmploymentType.FORMAL: (8.0, 0.4),
        EmploymentType.SELF_EMPLOYED: (6.0, 0.5),
        EmploymentType.INFORMAL: (4.0, 0.6),
        EmploymentType.UNEMPLOYED: (1.5, 0.7),
    }

    LINKED_BANK_PROB = {
        EmploymentType.FORMAL: 0.80,
        EmploymentType.SELF_EMPLOYED: 0.55,
        EmploymentType.INFORMAL: 0.25,
        EmploymentType.UNEMPLOYED: 0.10,
    }

    LATE_REPAYMENT_LAM = {
        EmploymentType.FORMAL: 0.3,
        EmploymentType.SELF_EMPLOYED: 1.0,
        EmploymentType.INFORMAL: 1.8,
        EmploymentType.UNEMPLOYED: 3.5,
    }

    NB_INCOME_LAM = {
        EmploymentType.FORMAL: 0.3,
        EmploymentType.SELF_EMPLOYED: 1.5,
        EmploymentType.INFORMAL: 1.2,
        EmploymentType.UNEMPLOYED: 0.2,
    }

    for i in range(n):
        employment = rd.choice(list(EmploymentType))

        age = rd.randint(20, 55)
        province = rd.choice(provinces_codes)
        nb_dependants = sample_dependants(age)

        median_income, sigma_income = INCOME_PARAMS[employment]
        mu = math.log(median_income) - (sigma_income**2) / 2
        income = sample_lognormal(mu, sigma_income)

        alpha, beta = RATIO_PARAMS[employment]
        ratio = sample_beta(alpha, beta)
        expenses = income * ratio

        median_mult, sigma_mult = DEBT_PARAMS[employment]
        mu_mult = math.log(median_mult) - (sigma_mult**2) / 2
        debt_mult = sample_lognormal(mu_mult, sigma_mult)
        existing_debt = income * debt_mult

        alpha_bill, beta_bill = BILL_PARAMS[employment]
        bill_payment_rate = sample_beta(alpha_bill, beta_bill)

        median_loan, sigma_loan = LOAN_PARAMS[employment]
        mu_loan = math.log(median_loan) - (sigma_loan**2) / 2
        loan_mult = sample_lognormal(mu_loan, sigma_loan)
        loan_amount = income * loan_mult

        p = INSURANCE_PARAMS[employment]
        has_social_insurance = rd.random() < p

        nb_income_sources = 1 + sample_poisson(NB_INCOME_LAM[employment])

        app_tenure_day = rd.randint(30, 1500)

        avg_tx_value_vnd = sample_lognormal(math.log(150_000), 0.8)
        avg_monthly_tx = (expenses / avg_tx_value_vnd) + rd.gauss(0, 2)
        avg_monthly_tx = max(1.0, avg_monthly_tx)

        has_linked_bank = rd.random() < LINKED_BANK_PROB[employment]

        late_repayment_count = sample_poisson(LATE_REPAYMENT_LAM[employment])

        median_topup, sigma_topup = TOPUP_PARAMS[employment]
        mu_topup = math.log(median_topup) - (sigma_topup**2) / 2
        wallet_top_up_freq = sample_lognormal(mu_topup, sigma_topup)

        dataset.append(
            CreditApplicant(
                applicant_id=f"VN_{i:05d}",
                age=age,
                province_code=province,
                nb_dependents=nb_dependants,
                monthly_income_vnd=income,
                monthly_expenses_vnd=expenses,
                existing_debt_vnd=existing_debt,
                loan_amount_requested_vnd=loan_amount,
                loan_duration_months=rd.randint(3, 36),
                employment_type=employment,
                tenure_months=rd.randint(0, 240),
                has_social_insurance=has_social_insurance,
                nb_income_sources=nb_income_sources,
                app_tenure_days=app_tenure_day,
                avg_monthly_transactions=avg_monthly_tx,
                avg_transaction_value_vnd=avg_tx_value_vnd,
                bill_payment_rate=bill_payment_rate,
                wallet_top_up_frequency=wallet_top_up_freq,
                has_linked_bank_account=has_linked_bank,
                late_repayment_count=late_repayment_count,
                loan_purpose=rd.choice(list(LoanPurpose)),
            )
        )

    return dataset


def compute_score(applicant: CreditApplicant, noise: Optional[float] = None):
    if noise is None:
        noise = 0.0
    late_norm = math.log(1 + applicant.late_repayment_count) / math.log(11)
    debt_ratio = min(
        applicant.existing_debt_vnd / (applicant.monthly_income_vnd + 1), 10.0
    )
    debt_norm = math.log(1 + debt_ratio) / math.log(11)
    expense_ratio = min(
        applicant.monthly_expenses_vnd / (applicant.monthly_income_vnd + 1), 10.0
    )
    expense_norm = math.log(1 + expense_ratio) / math.log(11)

    return (
        0.3 * debt_norm
        + 0.3 * late_norm
        + 0.1 * (applicant.employment_type == EmploymentType.UNEMPLOYED)
        + 0.1 * expense_norm
        + noise
    )


def assign_label(dataset):
    score_dataset = []
    for applicant in dataset:
        bruit = rd.gauss(0, 0.25)
        score_dataset.append(compute_score(applicant, bruit))

    score_dataset_sorted = sorted(score_dataset)
    theta = score_dataset_sorted[math.floor(len(score_dataset) * 0.85)]

    for applicant, score in zip(dataset, score_dataset):
        applicant.label = 1 if score > theta else 0

    return score_dataset, theta


dataset = generate_synthetic_dataset(1000)

# Diagnostic correct
score_dataset, theta = assign_label(dataset)

scores_formal = [
    s
    for a, s in zip(dataset, score_dataset)
    if a.employment_type == EmploymentType.FORMAL
]

print(f"Max FORMAL : {max(scores_formal):.3f}")
print(f"θ          : {theta:.3f}")
print(f"Écart      : {max(scores_formal) - theta:.3f}")
print(f"Écart      : {max(scores_formal) - theta:.3f}")

nb_defaut = sum(a.label for a in dataset)
print(f"Taux global : {nb_defaut / len(dataset):.1%}")

defaut_par_emploi = {}
for a in dataset:
    k = a.employment_type.value
    defaut_par_emploi.setdefault(k, []).append(a.label)

for k, labels in defaut_par_emploi.items():
    print(f"{k:15} → {sum(labels) / len(labels):.1%} de défauts")
