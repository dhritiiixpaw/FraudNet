
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# ENVIRONMENT / GEMINI
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="FraudNet | Financial Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DATA_FILE = ROOT_DIR / "data" / "raw" / "PaySim.csv"

REPORTS_FILE = RESULTS_DIR / "sample_sar_reports.csv"
GRAPH_FILE = RESULTS_DIR / "graph_flagged_transactions.csv"
SUSPICIOUS_FILE = RESULTS_DIR / "suspicious_accounts.csv"

SAMPLE_SIZE = 500_000


# --------------------------------------------------
# COLORS
# --------------------------------------------------

MAGENTA = "#D81B80"
PINK_BG = "#FCE7F3"
PINK_CARD = "#FFF8FC"
PLUM = "#552444"
PLUM_SOFT = "#754A68"
BORDER = "#F0C9DE"
RED = "#C6284D"


# --------------------------------------------------
# THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    .stApp {
        background: #FCE7F3;
        color: #43243A;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #F8D9E9;
        border-right: 1px solid #F0C9DE;
    }

    [data-testid="stSidebar"] * {
        color: #552444;
    }

    h1, h2, h3 {
        color: #552444 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    p, label {
        color: #43243A;
    }

    [data-testid="stMain"] .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: #FFF8FC;
        border: 1px solid #F0C9DE;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(85, 36, 68, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #754A68 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #D81B80 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }

    .stButton button {
        background: #D81B80;
        color: white;
        border: 1px solid #D81B80;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton button:hover {
        background: #B7156B;
        color: white;
        border-color: #B7156B;
    }

    .stDownloadButton button {
        background: #FFF8FC;
        color: #D81B80;
        border: 1px solid #D81B80;
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #F0C9DE;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        background: #FFF8FC;
        border: 1px solid #F0C9DE;
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #F0C9DE;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_csv(file_path):

    if file_path.exists():
        return pd.read_csv(file_path)

    return pd.DataFrame()


@st.cache_data
def load_dataset_sample(file_path, nrows):

    if file_path.exists():
        return pd.read_csv(file_path, nrows=nrows)

    return pd.DataFrame()


reports_df = load_csv(REPORTS_FILE)
graph_df = load_csv(GRAPH_FILE)
suspicious_df = load_csv(SUSPICIOUS_FILE)

dataset_df = load_dataset_sample(DATA_FILE, SAMPLE_SIZE)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

transaction_count = len(dataset_df)

if not dataset_df.empty and "isFraud" in dataset_df.columns:

    known_fraud_count = int(dataset_df["isFraud"].sum())

else:

    known_fraud_count = 0

non_fraud_count = transaction_count - known_fraud_count

report_count = len(reports_df)
graph_count = len(graph_df)

fraud_percentage = (
    (known_fraud_count / transaction_count) * 100
    if transaction_count > 0
    else 0
)


# --------------------------------------------------
# RISK SCORING FUNCTIONS
# --------------------------------------------------

def calculate_risk_score(transaction):

    transaction_amount = float(
        transaction.get("amount", 0)
    )

    transaction_type = str(
        transaction.get("type", "")
    ).upper()

    risk_score = 0

    # Transaction type risk
    if transaction_type in ["TRANSFER", "CASH_OUT"]:

        risk_score += 25

    # Transaction amount risk
    if transaction_amount >= 500_000:

        risk_score += 50

    elif transaction_amount >= 100_000:

        risk_score += 30

    elif transaction_amount >= 50_000:

        risk_score += 15

    risk_score = min(risk_score, 100)

    # Risk category
    if risk_score >= 60:

        risk_level = "High"

    elif risk_score >= 30:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    return risk_score, risk_level


def get_risk_reasons(transaction):

    reasons = []

    transaction_amount = float(
        transaction.get("amount", 0)
    )

    transaction_type = str(
        transaction.get("type", "")
    ).upper()

    if transaction_type in ["TRANSFER", "CASH_OUT"]:

        reasons.append(
            "Transaction type is TRANSFER or CASH_OUT"
        )

    if transaction_amount >= 500_000:

        reasons.append(
            "Very large transaction amount (₹500,000 or more)"
        )

    elif transaction_amount >= 100_000:

        reasons.append(
            "Large transaction amount (₹100,000 or more)"
        )

    elif transaction_amount >= 50_000:

        reasons.append(
            "Elevated transaction amount (₹50,000 or more)"
        )

    if not reasons:

        reasons.append(
            "No high-risk indicators identified by the demo rules"
        )

    return reasons


# --------------------------------------------------
# GEMINI AI INVESTIGATION FUNCTION
# --------------------------------------------------

def generate_ai_investigation(
    transaction,
    risk_score,
    risk_level,
):

    if client is None:

        raise ValueError(
            "Gemini API key was not found. "
            "Check your .env file."
        )

    transaction_details = {
        "transaction_type": str(
            transaction.get("type", "Unknown")
        ),
        "amount": float(
            transaction.get("amount", 0)
        ),
        "origin_account": str(
            transaction.get("nameOrig", "Unknown")
        ),
        "destination_account": str(
            transaction.get("nameDest", "Unknown")
        ),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reasons": get_risk_reasons(transaction),
    }

    prompt = f"""
You are FraudNet AI, an investigation assistant for a
financial fraud detection demonstration.

Analyze the following transaction information.

Transaction:
{transaction_details}

Write your response using these sections:

1. Transaction Overview
2. Potential Risk Indicators
3. Why This Transaction May Need Review
4. Recommended Investigation Steps
5. Important Limitations

Be concise and professional.

Do not claim that the transaction is confirmed fraud.
Do not invent information that is not provided.
Clearly distinguish risk indicators from confirmed facts.

The risk score is a heuristic demo score, not a validated
probability of fraud.

The dataset is synthetic PaySim data.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🛡️ FraudNet")

    st.caption("FINANCIAL INTELLIGENCE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Flagged Transactions",
            "Investigation Reports",
            "About FraudNet",
        ],
    )

    st.divider()

    st.caption("PROJECT STATUS")

    st.markdown("**DEMO ENVIRONMENT**")

    st.caption(
        "Synthetic transaction data · ML + graph investigation"
    )


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #FFF8FC, #FAD6E8);
        border: 1px solid #EFB8D4;
        padding: 30px 32px;
        border-radius: 20px;
        margin-bottom: 20px;
    ">
        <div style="
            color: #D81B80;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
        ">
            FRAUD DETECTION & INVESTIGATION PLATFORM
        </div>
        <div style="
            font-size: 36px;
            font-weight: 700;
            color: #552444;
            margin-top: 10px;
            line-height: 1.2;
        ">
            Financial Intelligence, Reimagined.
        </div>
        <div style="
            color: #754A68;
            font-size: 16px;
            line-height: 1.6;
            margin-top: 10px;
        ">
            Explore suspicious transaction patterns, machine learning
            predictions, and connected fraud signals through one
            investigation dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.caption("OVERVIEW")

    st.header("Fraud monitoring command center")

    st.write(
        "Monitor transaction activity, review investigation signals, "
        "and explore suspicious transaction patterns."
    )

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions analyzed",
        f"{transaction_count:,}",
    )

    col2.metric(
        "Known fraud cases",
        f"{known_fraud_count:,}",
        help="Fraud labels present in the loaded PaySim sample.",
    )

    col3.metric(
        "Investigation reports",
        f"{report_count:,}",
    )

    col4.metric(
        "Graph-linked transactions",
        f"{graph_count:,}",
    )

    # HIGH-RISK ALERT

    # MODEL ALERT

    st.error(
       "🔴 MODEL ALERT — TRANSACTIONS FLAGGED\n\n"
       f"{report_count:,} transactions flagged by the Random Forest "
       "model for further investigation.\n\n"
       "These are model predictions, not confirmed fraud cases. "
       "The alert count represents model-flagged transactions "
       "with generated sample reports."
)

    # MODEL PERFORMANCE

    st.divider()

    st.subheader("Model Performance")

    st.caption(
        "Random Forest evaluation on the held-out test dataset. "
        "Metrics are based on the current training run."
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("Accuracy", "99.98%")

    with metric2:
        st.metric("Fraud Precision", "92.59%")

    with metric3:
        st.metric("Fraud Recall", "53.19%")

    with metric4:
        st.metric("Fraud Detected", "25 / 47")

    st.info(
        "Fraud recall is especially important: it measures how many "
        "known fraud transactions the model successfully detected. "
        "These results are from a limited sample and are not "
        "representative of real-world banking performance."
    )

    # RISK DISTRIBUTION

    st.divider()

    st.subheader("Risk distribution")

    st.caption(
        "Distribution of graph-linked transactions by heuristic risk level."
    )

    if not graph_df.empty:

        risk_results = graph_df.apply(
            calculate_risk_score,
            axis=1,
        )

        graph_df["risk_score"] = [
            result[0] for result in risk_results
        ]

        graph_df["risk_level"] = [
            result[1] for result in risk_results
        ]

        risk_distribution = (
            graph_df["risk_level"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0,
            )
            .to_frame("Transactions")
        )

        st.bar_chart(
            risk_distribution,
            color=MAGENTA,
        )

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        risk_col1.metric(
            "High risk",
            f"{(graph_df['risk_level'] == 'High').sum():,}",
        )

        risk_col2.metric(
            "Medium risk",
            f"{(graph_df['risk_level'] == 'Medium').sum():,}",
        )

        risk_col3.metric(
            "Low risk",
            f"{(graph_df['risk_level'] == 'Low').sum():,}",
        )

    else:

        st.info(
            "No graph-linked transactions available for risk analysis."
        )

    # TRANSACTION INTELLIGENCE

    st.subheader("Transaction intelligence")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Transaction activity")

        st.caption(
            "How transactions are distributed by transaction type."
        )

        if not dataset_df.empty and "type" in dataset_df.columns:

            transaction_types = (
                dataset_df["type"]
                .value_counts()
                .rename_axis("Transaction type")
                .to_frame("Transactions")
            )

            st.bar_chart(
                transaction_types,
                color=MAGENTA,
            )

        else:

            st.info("Transaction type data is unavailable.")

    with right:

        st.markdown("#### Fraud label distribution")

        st.caption(
            "Known fraud labels versus non-fraud labels in the sample."
        )

        if transaction_count > 0:

            fraud_distribution = pd.DataFrame(
                {
                    "Label": ["Non-fraud", "Fraud"],
                    "Count": [
                        non_fraud_count,
                        known_fraud_count,
                    ],
                }
            ).set_index("Label")

            st.bar_chart(
                fraud_distribution,
                color="#E88DBB",
            )

            st.caption(
                f"Fraud represents {fraud_percentage:.3f}% "
                "of the loaded sample."
            )

        else:

            st.info("Dataset is unavailable.")

    # INVESTIGATION QUEUE

    st.divider()

    st.subheader("Investigation queue")

    st.write(
        "Graph-linked transactions available for further review."
    )

    if not graph_df.empty:

        queue_col1, queue_col2 = st.columns([3, 1])

        with queue_col1:

            st.metric(
                "Transactions awaiting review",
                f"{graph_count:,}",
            )

        with queue_col2:

            st.download_button(
                label="Export queue",
                data=graph_df.to_csv(index=False),
                file_name="fraudnet_investigation_queue.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            graph_df.head(10),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "These records are graph-linked investigation signals. "
            "They are not independent graph-model predictions."
        )

    else:

        st.info(
            "No investigation queue found. "
            "Run src/graph_detection.py first."
        )

    # RECENT REPORTS

    st.divider()

    st.subheader("Recent investigation reports")

    if not reports_df.empty:

        st.dataframe(
            reports_df.head(5),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Automatically generated sample reports for model-flagged "
            "transactions."
        )

    else:

        st.info(
            "No reports found. Run src/generate_reports.py first."
        )

    st.divider()

    st.caption(
        "FraudNet is a prototype using synthetic PaySim data. "
        "Metrics describe the loaded sample, not live banking activity."
    )


# --------------------------------------------------
# FLAGGED TRANSACTIONS PAGE
# --------------------------------------------------

elif page == "Flagged Transactions":

    st.caption("TRANSACTION MONITORING")

    st.header("Investigation queue")

    st.write(
        "Review graph-linked transactions and inspect individual "
        "records to understand their available details."
    )

    if not graph_df.empty:

        st.metric(
            "Graph-linked transactions",
            f"{len(graph_df):,}",
        )

        # ------------------------------------------
        # RISK FILTER
        # ------------------------------------------

        filtered_df = graph_df.copy()

        risk_results = filtered_df.apply(
            calculate_risk_score,
            axis=1,
        )

        filtered_df["risk_score"] = [
            result[0] for result in risk_results
        ]

        filtered_df["risk_level"] = [
            result[1] for result in risk_results
        ]

        selected_risk = st.selectbox(
            "Filter by risk level",
            ["All", "High", "Medium", "Low"],
        )

        if selected_risk != "All":

            filtered_df = filtered_df[
                filtered_df["risk_level"] == selected_risk
            ]

        st.caption(
            f"Showing {len(filtered_df):,} of {len(graph_df):,} "
            "graph-linked transactions."
        )

        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True,
        )

        # ------------------------------------------
        # NETWORK VISUALIZATION
        # ------------------------------------------

        st.divider()

        st.subheader("Fraud network visualization")

        st.write(
    "Explore account relationships among transactions linked "
    "to known fraud labels in the synthetic dataset. "
    "This visualization demonstrates network-based investigation, "
    "not independent graph-model predictions."
)

        if (
            "nameOrig" in graph_df.columns
            and "nameDest" in graph_df.columns
        ):

            network_df = filtered_df[
                ["nameOrig", "nameDest"]
            ].dropna().head(10)

            if not network_df.empty:

                dot = """
                graph FraudNetwork {
                    bgcolor="#FFF8FC";
                    rankdir=LR;
                    overlap=false;
                    splines=true;

                    node [
                        style=filled,
                        fontname="Arial",
                        fontsize=10,
                        color="#D81B80",
                        fillcolor="#FAD6E8",
                        fontcolor="#552444"
                    ];

                    edge [
                        color="#D81B80",
                        penwidth=1.5
                    ];
                """

                for _, row in network_df.iterrows():

                    source = str(row["nameOrig"]).replace('"', "")
                    target = str(row["nameDest"]).replace('"', "")

                    dot += f'"{source}" -- "{target}";\n'

                dot += "}"

                st.graphviz_chart(
                    dot,
                    use_container_width=True,
                )

                st.caption(
    "Network preview of up to 10 graph-linked transactions "
    "matching the selected risk filter. Connections are based "
    "on known dataset fraud labels and are not independent "
    "graph-model predictions."
)
            else:

                st.info(
                    "No network connections match the selected risk filter."
                )

        else:

            st.warning(
                "The graph file does not contain the expected "
                "account columns (nameOrig and nameDest)."
            )

        # ------------------------------------------
        # DOWNLOAD FLAGGED TRANSACTIONS
        # ------------------------------------------

        st.download_button(
            label="Download flagged transactions",
            data=filtered_df.to_csv(index=False),
            file_name="fraudnet_flagged_transactions.csv",
            mime="text/csv",
        )

        # ------------------------------------------
        # TRANSACTION INVESTIGATION
        # ------------------------------------------

        st.divider()

        st.subheader("Transaction investigation")

        if not filtered_df.empty:

            selected_row = st.selectbox(
                "Select a transaction to investigate",
                options=list(range(len(filtered_df))),
                format_func=lambda i: (
                    f"Transaction {i + 1} — "
                    f"{filtered_df.iloc[i]['risk_level']} Risk"
                ),
            )

            selected_transaction = filtered_df.iloc[selected_row]

            # Calculate score BEFORE using it
            risk_score, risk_level = calculate_risk_score(
                selected_transaction
            )

            # Unique transaction key for AI report
            transaction_id = str(
                selected_transaction.get(
                    "nameOrig",
                    selected_row,
                )
            )

            ai_report_key = f"ai_report_{transaction_id}"

            # ------------------------------------------
            # AI INVESTIGATION COPILOT
            # ------------------------------------------

            st.divider()

            st.markdown(
                "### 🤖 FraudNet AI Investigation Copilot"
            )

            st.write(
                "Generate an AI-powered explanation of this transaction "
                "and receive recommendations for further investigation."
            )

            if st.button(
                "✨ Generate AI Investigation",
                key=f"generate_ai_{selected_row}",
                width="stretch",
            ):

                if not GEMINI_API_KEY:

                    st.error(
                        "Gemini API key not found. "
                        "Please check your .env file."
                    )

                else:

                    with st.spinner(
                        "FraudNet AI is analyzing the transaction..."
                    ):

                        try:

                            ai_report = generate_ai_investigation(
                                selected_transaction,
                                risk_score,
                                risk_level,
                            )

                            st.session_state[ai_report_key] = ai_report

                        except Exception as e:

                            st.error(
                                f"AI investigation failed: {e}"
                            )

            # ------------------------------------------
            # DISPLAY AI REPORT
            # ------------------------------------------

            if st.session_state.get(ai_report_key):

                st.markdown("#### AI Investigation Report")

                st.markdown(
                    st.session_state[ai_report_key]
                )

                st.caption(
                    "AI-generated analysis for investigation assistance. "
                    "Not a confirmed fraud finding."
                )

                st.download_button(
                    label="Download AI Investigation",
                    data=st.session_state[ai_report_key],
                    file_name=(
                        f"fraudnet_ai_investigation_{selected_row}.txt"
                    ),
                    mime="text/plain",
                    key=f"download_ai_{selected_row}",
                    width="stretch",
                )

            # ------------------------------------------
            # ACCOUNT-LEVEL INVESTIGATION
            # ------------------------------------------

            st.divider()

            st.markdown("### Account-level investigation")

            origin_account = str(
                selected_transaction.get(
                    "nameOrig",
                    "Unknown",
                )
            )

            destination_account = str(
                selected_transaction.get(
                    "nameDest",
                    "Unknown",
                )
            )

            account_col1, account_col2 = st.columns(2)

            with account_col1:

                st.markdown("#### Origin account")

                st.code(origin_account)

                origin_activity = graph_df[
                    (graph_df["nameOrig"] == origin_account)
                    | (graph_df["nameDest"] == origin_account)
                ]

                st.metric(
                    "Connected transactions",
                    f"{len(origin_activity):,}",
                )

            with account_col2:

                st.markdown("#### Destination account")

                st.code(destination_account)

                destination_activity = graph_df[
                    (graph_df["nameOrig"] == destination_account)
                    | (graph_df["nameDest"] == destination_account)
                ]

                st.metric(
                    "Connected transactions",
                    f"{len(destination_activity):,}",
                )

            # ------------------------------------------
            # RELATED TRANSACTIONS
            # ------------------------------------------

            st.markdown("### Related transaction activity")

            related_transactions = graph_df[
                (
                    graph_df["nameOrig"].isin(
                        [origin_account, destination_account]
                    )
                )
                |
                (
                    graph_df["nameDest"].isin(
                        [origin_account, destination_account]
                    )
                )
            ].copy()

            if not related_transactions.empty:

                st.caption(
                    f"Found {len(related_transactions):,} graph-linked "
                    "transactions involving either account."
                )

                st.dataframe(
                    related_transactions,
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.info(
                    "No additional connected transactions were found."
                )

            st.caption(
                "Account activity is limited to records available in "
                "the graph-linked dataset."
            )

            # ------------------------------------------
            # DOWNLOAD INVESTIGATION SUMMARY
            # ------------------------------------------

            st.divider()

            st.markdown("### Export investigation")

            amount = float(
                selected_transaction.get("amount", 0)
            )

            transaction_type = str(
                selected_transaction.get("type", "Unknown")
            )

            report_text = f"""
FRAUDNET
Financial Intelligence & Investigation Platform

TRANSACTION INVESTIGATION SUMMARY
--------------------------------

Transaction Details
------------------
Origin Account: {origin_account}
Destination Account: {destination_account}
Transaction Type: {transaction_type}
Transaction Amount: {amount:,.2f}

Risk Assessment
--------------
Risk Score: {risk_score}/100
Risk Level: {risk_level}

Account Activity
---------------
Origin Account Connected Transactions:
{len(origin_activity)}

Destination Account Connected Transactions:
{len(destination_activity)}

Related Transactions:
{len(related_transactions)}

Investigation Notes
------------------
This report is generated from a synthetic PaySim dataset.

Risk scores are heuristic indicators for demonstration purposes.
Model predictions and graph-linked records are not confirmed
fraud findings.

This report is not an official Suspicious Activity Report (SAR).
"""

            st.download_button(
                label="Download Investigation Summary",
                data=report_text,
                file_name=(
                    f"fraudnet_investigation_{selected_row}.txt"
                ),
                mime="text/plain",
                key=f"download_summary_{selected_row}",
                width="stretch",
            )

            # ------------------------------------------
            # RISK SCORE DISPLAY
            # ------------------------------------------

            st.markdown("### Transaction risk assessment")

            risk_col1, risk_col2 = st.columns(2)

            with risk_col1:

                st.metric(
                    "Demo risk score",
                    f"{risk_score} / 100",
                )

            with risk_col2:

                st.metric(
                    "Risk level",
                    risk_level,
                )

            st.progress(risk_score / 100)

            st.caption(
                "Heuristic score based on transaction type and amount. "
                "This is not an ML prediction or a confirmed fraud verdict."
            )

            # ------------------------------------------
            # RISK REASONS
            # ------------------------------------------

            st.markdown("### Why was this transaction flagged?")

            risk_reasons = get_risk_reasons(
                selected_transaction
            )

            for reason in risk_reasons:

                st.write(f"• {reason}")

            # ------------------------------------------
            # TRANSACTION DETAILS
            # ------------------------------------------

            st.divider()

            st.markdown("### Transaction details")

            details_col, status_col = st.columns([2, 1])

            with details_col:

                st.dataframe(
                    selected_transaction.to_frame(
                        name="Value"
                    ),
                    width="stretch",
                )

            with status_col:

                st.markdown("### Investigation status")

                status_key = f"status_{selected_row}"
                note_key = f"note_{selected_row}"

                if status_key not in st.session_state:

                    st.session_state[status_key] = "Open"

                if note_key not in st.session_state:

                    st.session_state[note_key] = ""

                status = st.selectbox(
                    "Status",
                    ["Open", "Under Review", "Closed"],
                    key=status_key,
                )

                note = st.text_area(
                    "Investigation notes",
                    placeholder="Add observations or next steps...",
                    key=note_key,
                )

                st.caption(
                    "Status and notes are temporary and reset when "
                    "the app session is restarted."
                )

            st.info(
                "This record is a graph-linked investigation signal "
                "based on known fraud labels in the dataset. "
                "It is not an independent fraud prediction."
            )

        else:

            st.warning(
                "No transactions match the selected risk filter. "
                "Choose another risk level to continue."
            )

    else:

        st.info(
            "No graph-linked transaction file found. "
            "Run src/graph_detection.py first."
        )

    # ------------------------------------------
    # SUSPICIOUS ACCOUNTS
    # ------------------------------------------

    st.divider()

    st.subheader("Suspicious accounts")

    if not suspicious_df.empty:

        st.dataframe(
            suspicious_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No suspicious account file found."
        )


# --------------------------------------------------
# INVESTIGATION REPORTS PAGE
# --------------------------------------------------

elif page == "Investigation Reports":

    st.caption("CASE MANAGEMENT")

    st.header("Investigation reports")

    st.write(
        "Review and export sample reports generated from "
        "model-flagged transactions."
    )

    if not reports_df.empty:

        st.metric(
            "Reports generated",
            f"{len(reports_df):,}",
        )

        st.dataframe(
            reports_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            label="Download investigation reports",
            data=reports_df.to_csv(index=False),
            file_name="fraudnet_sample_investigation_reports.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("View a report")

        selected_report = st.selectbox(
            "Select a report",
            options=list(range(len(reports_df))),
            format_func=lambda i: f"Report {i + 1}",
        )

        report_details = reports_df.iloc[
            selected_report
        ].to_frame(name="Value")

        st.dataframe(
            report_details,
            width="stretch",
        )

    else:

        st.info(
            "No reports found. Run src/generate_reports.py first."
        )

    st.divider()

    st.subheader("Report disclaimer")

    st.write(
        "These are automatically generated sample investigation reports. "
        "They are not official Suspicious Activity Reports (SARs), and "
        "they have not been reviewed by a compliance officer."
    )


# --------------------------------------------------
# ABOUT FRAUDNET PAGE
# --------------------------------------------------

elif page == "About FraudNet":

    st.caption("PROJECT INFORMATION")

    st.header("About FraudNet")

    st.write(
        "FraudNet is a prototype financial fraud detection and "
        "investigation platform built to explore how machine learning "
        "and graph analysis can support suspicious transaction monitoring."
    )

    st.subheader("Core components")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("01 · Machine learning")

            st.write(
                "A Random Forest classifier is trained on transaction-level "
                "features to identify potentially suspicious activity."
            )

        with st.container(border=True):

            st.subheader("02 · Graph investigation")

            st.write(
                "Account relationships are explored to demonstrate how "
                "transaction networks can support fraud investigations."
            )

    with col2:

        with st.container(border=True):

            st.subheader("03 · Investigation reports")

            st.write(
                "Sample reports organize model-flagged transactions into "
                "a format that can support further manual review."
            )

        with st.container(border=True):

            st.subheader("04 · AI investigation copilot")

            st.write(
                "Gemini generates explanatory narratives and investigation "
                "recommendations from selected transaction details."
            )

    st.subheader("Technology stack")

    st.code(
        "Python · Pandas · Scikit-learn · NetworkX · Streamlit · Gemini API",
        language="text",
    )

    st.subheader("Dataset")

    st.write(
        "FraudNet currently uses a sample of the PaySim synthetic "
        "mobile money transaction dataset."
    )

    st.markdown(
        """
        - Up to 500,000 transaction records are loaded in the dashboard
        - Fraud labels are from the synthetic dataset
        - No actual customer or banking records are used
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - This is a prototype, not a production fraud detection system.
        - It does not connect to real banks or payment networks.
        - Graph-linked investigation results are not independent predictions.
        - Model performance requires further evaluation and validation.
        - AI-generated explanations may be incomplete or inaccurate.
        - Sample reports are not official regulatory filings.
        """
    )

    st.divider()

    st.caption(
        "FraudNet · Financial Intelligence Prototype"
    )
