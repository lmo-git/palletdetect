"""
PalletVision — AI Pallet Counting App
Streamlit Gradient Blue Design + Gemini AI Counting

Run:
pip install "streamlit>=1.51" pandas pillow openpyxl google-genai
streamlit run pallet_vision_app.py
"""

import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PalletVision",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================================================
# GEMINI API KEY FROM STREAMLIT SECRETS
# =========================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is empty. Please check Streamlit Secrets.")
        st.stop()

    client = genai.Client(api_key=GEMINI_API_KEY)

except KeyError:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.info(
        """
        Please add this in Streamlit Cloud > App settings > Secrets:

        GEMINI_API_KEY = "your_gemini_api_key_here"
        """
    )
    st.stop()

except Exception as e:
    st.error(f"Gemini client initialization error: {type(e).__name__}: {e}")
    st.stop()


# =========================================================
# PALLET COUNTING PROMPT
# =========================================================
PALLET_PROMPT = """
You are an expert AI visual inspector specializing in transport pallet logistics and inventory verification.

You will receive exactly two images of the same truck load:
1. Side view of the truck
2. Rear view of the truck

[GOAL]
Accurately count the ACTUAL TOTAL number of transport pallets loaded on the truck by combining 3D spatial evidence from both images.

[CRITICAL COUNTING LOGIC & DEFINITIONS]

Total Pallets = Height Layers × Width Columns × Depth Rows

1. Width Columns:
   - Identify mainly from the REAR view.
   - Count how many distinct vertical pallet stacks sit side-by-side across the truck width.
   - Example: left stack and right stack = 2 width columns.

2. Depth Rows:
   - Identify mainly from the SIDE view.
   - Count how many pallet stack positions extend along the truck length from rear to front.
   - Example: only one stack position at the tail = 1 depth row.

3. Height Layers:
   - Double-check from BOTH side view and rear view.
   - Count every individual pallet vertically from bottom to top.
   - Do NOT only count large plastic blocks.
   - Look carefully for:
     - forklift entry gaps
     - horizontal seams
     - lip-to-lip contact lines
     - nested pallet feet
     - stacked pallet edges
   - If pallets are nested or interlocked, still count each individual pallet.
   - If user hint or ground truth specifies the exact structural height layer count, use it to calibrate the visual layer count.

[STEP-BY-STEP INSTRUCTION]

Step 1:
Analyze the REAR view to determine Width Columns.

Step 2:
Analyze the SIDE view to determine Depth Rows and confirm cargo position.

Step 3:
Zoom in mentally on the stack in BOTH views.
Count the exact number of individual pallets vertically as Height Layers.

Step 4:
Calculate:
Total Pallets = Height Layers × Width Columns × Depth Rows

[IMPORTANT RULES]

- Count only actual transport pallets.
- Do not count truck body, doors, floor, straps, shadows, wheels, license plates, background, boxes, documents, or empty truck space.
- Do not use a fixed default quantity.
- Do not assume 12, 28, or any fixed number unless the image structure supports it.
- The side view and rear view are two views of the same truck load.
- Do not double-count the same stack shown from different angles.
- Final pallet count must be an integer.
- Always set needHumanReview to true.
- If all visible pallets are red, classify palletColor as red.
- If multiple colors are visible, separate the result by color.
- If color is unclear, use unknown.

[OUTPUT REQUIREMENT]

Return STRICT JSON only.
Do not return markdown.
Do not return explanation outside JSON.

JSON shape:
{
  "ok": true,
  "confidence": "high|medium|low",
  "fileName": {
    "sideViewImage": "",
    "rearViewImage": "",
    "combinedImageName": ""
  },
  "palletColor": "red|blue|white|wood|black|green|other|unknown",
  "countingExplanation": {
    "heightLayers": 0,
    "heightLayersExplanation": "",
    "widthColumns": 0,
    "widthColumnsExplanation": "",
    "depthRows": 0,
    "depthRowsExplanation": "",
    "formula": "",
    "calculationMethod": ""
  },
  "summary": {
    "totalPallets": 0,
    "assumptions": "",
    "riskOfError": "low|medium|high",
    "needHumanReview": true
  },
  "resultRows": [
    {
      "imageFileName": "",
      "palletColor": "red|blue|white|wood|black|green|other|unknown",
      "palletCount": 0
    }
  ]
}
"""


# =========================================================
# CSS DESIGN
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 0 !important;
    max-width: 840px;
}

.topbar {
    background: linear-gradient(135deg, #042C53 0%, #185FA5 60%, #378ADD 100%);
    padding: 14px 24px;
    border-radius: 0 0 16px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 22px;
}

.topbar-logo {
    color: #fff;
    font-size: 20px;
    font-weight: 600;
}

.topbar-badge {
    background: rgba(255,255,255,0.18);
    color: #fff;
    font-size: 12px;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.30);
}

.stepbar {
    background: linear-gradient(90deg, #0C447C, #185FA5);
    padding: 11px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 22px;
    font-size: 13px;
    color: rgba(255,255,255,0.60);
}

.step-active {
    color: #fff;
    font-weight: 500;
}

.step-done {
    background: #378ADD;
    color: #fff;
    width: 21px;
    height: 21px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
}

.step-circle-active {
    background: #fff;
    color: #185FA5;
    width: 21px;
    height: 21px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
}

.step-circle {
    background: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.70);
    width: 21px;
    height: 21px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
}

.step-sep {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.25);
}

.upload-label-box {
    background: linear-gradient(135deg, #E6F1FB, #B5D4F4);
    border: 2px solid #185FA5;
    border-radius: 14px;
    padding: 13px 16px;
    text-align: center;
    color: #0C447C;
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 6px;
}

.result-header {
    background: linear-gradient(135deg, #042C53 0%, #185FA5 100%);
    border-radius: 14px 14px 0 0;
    padding: 18px 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.result-title {
    color: #fff;
    font-size: 16px;
    font-weight: 600;
}

.result-sub {
    color: #B5D4F4;
    font-size: 12px;
    margin-top: 4px;
}

.result-num {
    color: #fff;
    font-size: 42px;
    font-weight: 600;
    text-align: right;
    line-height: 1;
}

.result-num-label {
    color: #B5D4F4;
    font-size: 12px;
    text-align: right;
    margin-top: 3px;
}

.info-card {
    background: white;
    border-left: 1px solid #B5D4F4;
    border-right: 1px solid #B5D4F4;
    padding: 16px 22px;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-top: 8px;
}

.metric-box {
    background: #E6F1FB;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}

.metric-label {
    color: #5F5E5A;
    font-size: 12px;
}

.metric-value {
    color: #185FA5;
    font-size: 22px;
    font-weight: 600;
    margin-top: 3px;
}

.total-bar {
    background: #F0F4F8;
    border: 1px solid #B5D4F4;
    border-radius: 0 0 14px 14px;
    padding: 12px 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.total-label {
    color: #5F5E5A;
    font-size: 13px;
}

.total-num {
    color: #185FA5;
    font-size: 21px;
    font-weight: 600;
}

.note-box {
    background: #E6F1FB;
    border-left: 4px solid #378ADD;
    border-radius: 0 10px 10px 0;
    padding: 11px 15px;
    font-size: 13px;
    color: #0C447C;
    margin: 14px 0;
}

.toast-success {
    background: linear-gradient(135deg, #0F6E56, #1D9E75);
    color: #fff;
    border-radius: 12px;
    padding: 13px 18px;
    font-size: 14px;
    font-weight: 500;
    margin-top: 12px;
}

.error-box {
    background: #FDEAEA;
    border-left: 4px solid #E24B4A;
    border-radius: 0 10px 10px 0;
    padding: 11px 15px;
    font-size: 13px;
    color: #8A1F1F;
    margin: 14px 0;
}

div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 11px 18px !important;
}

div[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
PALLET_TYPE_OPTIONS = ["Chep", "Loscam", "Other"]
PALLET_TYPE_DATA_DICT = {
    "red": "Loscam",
    "blue": "Chep",
}
PALLET_TYPE_DEFAULT = "Other"

CURRENT_COLUMNS = ["imageFileName", "palletColor", "palletType", "palletCount"]
BATCH_COLUMNS = [
    "transactionNo",
    "transactionKey",
    "savedAt",
    "reviewerNote",
    "sideViewImage",
    "rearViewImage",
    "confidence",
    "riskOfError",
    "imageFileName",
    "palletColor",
    "palletType",
    "palletCount",
]

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "result_df" not in st.session_state:
    st.session_state.result_df = pd.DataFrame(columns=CURRENT_COLUMNS)

if "raw_json" not in st.session_state:
    st.session_state.raw_json = ""

if "batch_transactions" not in st.session_state:
    st.session_state.batch_transactions = []

if "batch_df" not in st.session_state:
    st.session_state.batch_df = pd.DataFrame(columns=BATCH_COLUMNS)

if "transaction_no" not in st.session_state:
    st.session_state.transaction_no = 1

if "widget_version" not in st.session_state:
    st.session_state.widget_version = 0

if "download_json" not in st.session_state:
    st.session_state.download_json = None

if "download_excel" not in st.session_state:
    st.session_state.download_excel = None

if "download_total" not in st.session_state:
    st.session_state.download_total = 0

if "download_transaction_count" not in st.session_state:
    st.session_state.download_transaction_count = 0

if "flash_message" not in st.session_state:
    st.session_state.flash_message = ""

FOLDER_RESULT_COLUMNS = [
    "folderTransaction",
    "sideViewImage",
    "rearViewImage",
    "confidence",
    "riskOfError",
    "imageFileName",
    "palletColor",
    "palletType",
    "palletCount",
]

if "folder_result_df" not in st.session_state:
    st.session_state.folder_result_df = pd.DataFrame(columns=FOLDER_RESULT_COLUMNS)

if "folder_ai_results" not in st.session_state:
    st.session_state.folder_ai_results = {}

if "folder_analysis_errors" not in st.session_state:
    st.session_state.folder_analysis_errors = []

if "folder_widget_version" not in st.session_state:
    st.session_state.folder_widget_version = 0

# Session-state migration when the app is refreshed after adding transactionKey.
if "transactionKey" not in st.session_state.batch_df.columns:
    st.session_state.batch_df.insert(
        1,
        "transactionKey",
        [f"Transaction #{value}" for value in st.session_state.batch_df.get("transactionNo", [])],
    )


# Session-state migration for Pallet Type when users refresh from an older app version.
def _migrate_pallet_type_state(df, expected_columns):
    migrated = pd.DataFrame(df).copy()
    if "palletColor" not in migrated.columns:
        migrated["palletColor"] = "unknown"

    default_type = (
        migrated["palletColor"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(PALLET_TYPE_DATA_DICT)
        .fillna(PALLET_TYPE_DEFAULT)
    )

    if "palletType" not in migrated.columns:
        migrated["palletType"] = default_type
    else:
        normalized_type = (
            migrated["palletType"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map({option.lower(): option for option in PALLET_TYPE_OPTIONS})
        )
        migrated["palletType"] = normalized_type.fillna(default_type)

    for column in expected_columns:
        if column not in migrated.columns:
            migrated[column] = 0 if column == "palletCount" else ""

    return migrated[expected_columns]


st.session_state.result_df = _migrate_pallet_type_state(
    st.session_state.result_df,
    CURRENT_COLUMNS,
)
st.session_state.batch_df = _migrate_pallet_type_state(
    st.session_state.batch_df,
    BATCH_COLUMNS,
)
st.session_state.folder_result_df = _migrate_pallet_type_state(
    st.session_state.folder_result_df,
    FOLDER_RESULT_COLUMNS,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def now_bangkok():
    return datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M:%S")


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def map_pallet_type(pallet_color):
    """Map the AI-detected pallet color to the default business pallet type."""
    normalized_color = str(pallet_color or "unknown").strip().lower()
    return PALLET_TYPE_DATA_DICT.get(normalized_color, PALLET_TYPE_DEFAULT)


def ensure_pallet_type(df, refresh_from_color=False):
    """Add/normalize Pallet Type while preserving a valid user override."""
    typed_df = pd.DataFrame(df).copy()

    if "palletColor" not in typed_df.columns:
        typed_df["palletColor"] = "unknown"

    default_type = typed_df["palletColor"].apply(map_pallet_type)

    if "palletType" not in typed_df.columns or refresh_from_color:
        typed_df["palletType"] = default_type
        return typed_df

    normalized_type = (
        typed_df["palletType"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map({option.lower(): option for option in PALLET_TYPE_OPTIONS})
    )
    typed_df["palletType"] = normalized_type.fillna(default_type)
    return typed_df


def safe_uploaded_name(upload, fallback_name):
    file_name = getattr(upload, "name", "") or fallback_name

    # Camera input commonly returns a generic file name. Rename it by transaction
    # so that side and rear images are distinguishable in the exported result.
    if file_name.lower() in {"camera_image.jpg", "camera_image.jpeg", "camera_image.png"}:
        return fallback_name

    return file_name


def normalize_result_rows(result, combined_file_name):
    summary = result.get("summary", {})
    explanation = result.get("countingExplanation", {})

    height = safe_int(explanation.get("heightLayers", 0))
    width = safe_int(explanation.get("widthColumns", 0))
    depth = safe_int(explanation.get("depthRows", 0))

    calculated_total = height * width * depth
    ai_total = safe_int(summary.get("totalPallets", calculated_total))

    rows = result.get("resultRows", [])

    if not rows:
        rows = [
            {
                "imageFileName": combined_file_name,
                "palletColor": result.get("palletColor", "unknown"),
                "palletCount": ai_total,
            }
        ]

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=CURRENT_COLUMNS)

    if "imageFileName" not in df.columns:
        df["imageFileName"] = combined_file_name

    if "palletColor" not in df.columns:
        df["palletColor"] = result.get("palletColor", "unknown")

    if "palletCount" not in df.columns:
        df["palletCount"] = ai_total

    df["imageFileName"] = df["imageFileName"].fillna("").replace("", combined_file_name)
    df["palletColor"] = df["palletColor"].fillna("unknown").replace("", "unknown")
    df = ensure_pallet_type(df, refresh_from_color=True)
    df["palletCount"] = pd.to_numeric(
        df["palletCount"],
        errors="coerce",
    ).fillna(0).astype(int)

    return df[CURRENT_COLUMNS]


def analyze_pallets_with_gemini(side_upload, rear_upload, hint, transaction_no):
    side_file_name = safe_uploaded_name(
        side_upload,
        f"transaction_{transaction_no:03d}_side.jpg",
    )
    rear_file_name = safe_uploaded_name(
        rear_upload,
        f"transaction_{transaction_no:03d}_rear.jpg",
    )
    combined_file_name = f"{side_file_name} + {rear_file_name}"

    # camera_input and file_uploader both return file-like UploadedFile objects.
    side_upload.seek(0)
    rear_upload.seek(0)
    side_img = Image.open(side_upload)
    rear_img = Image.open(rear_upload)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PALLET_PROMPT,
            f"Transaction number: {transaction_no}",
            f"Side view image filename: {side_file_name}",
            f"Rear view image filename: {rear_file_name}",
            f"Combined image filename: {combined_file_name}",
            f"Additional user hint: {hint or 'No additional context.'}",
            "Image angle: side view",
            side_img,
            "Image angle: rear view",
            rear_img,
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text

    try:
        result = json.loads(raw_text)
    except Exception:
        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        result = json.loads(cleaned)
        raw_text = cleaned

    # Always retain the real filenames used by this transaction, even when the
    # model returns blank or altered filename fields.
    result.setdefault("fileName", {})
    result["fileName"]["sideViewImage"] = side_file_name
    result["fileName"]["rearViewImage"] = rear_file_name
    result["fileName"]["combinedImageName"] = combined_file_name

    df = normalize_result_rows(result, combined_file_name)

    return result, df, raw_text


def validate_result_df(df):
    cleaned_df = pd.DataFrame(df).copy()

    for col in CURRENT_COLUMNS:
        if col not in cleaned_df.columns:
            raise ValueError(f"Missing required column: {col}")

    cleaned_df["imageFileName"] = cleaned_df["imageFileName"].fillna("").astype(str)
    cleaned_df["palletColor"] = cleaned_df["palletColor"].fillna("unknown").astype(str)
    cleaned_df = ensure_pallet_type(cleaned_df)
    cleaned_df["palletCount"] = pd.to_numeric(
        cleaned_df["palletCount"],
        errors="coerce",
    ).fillna(0).astype(int)

    return cleaned_df[CURRENT_COLUMNS]


def build_transaction(df, reviewer_note, ai_result, transaction_no, transaction_key=None):
    cleaned_df = validate_result_df(df)
    saved_at = now_bangkok()
    result = ai_result or {}
    summary = result.get("summary", {})
    file_names = result.get("fileName", {})
    final_total = int(cleaned_df["palletCount"].sum())

    transaction = {
        "transactionNo": int(transaction_no),
        "transactionKey": transaction_key or f"Transaction #{int(transaction_no)}",
        "savedAt": saved_at,
        "reviewerNote": reviewer_note or "",
        "finalTotalPallets": final_total,
        "confidence": result.get("confidence", "-"),
        "riskOfError": summary.get("riskOfError", "-"),
        "sideViewImage": file_names.get("sideViewImage", ""),
        "rearViewImage": file_names.get("rearViewImage", ""),
        "resultRows": cleaned_df.to_dict(orient="records"),
        "aiRawResult": result,
    }

    flat_df = cleaned_df.copy()
    flat_df.insert(0, "riskOfError", transaction["riskOfError"])
    flat_df.insert(0, "confidence", transaction["confidence"])
    flat_df.insert(0, "rearViewImage", transaction["rearViewImage"])
    flat_df.insert(0, "sideViewImage", transaction["sideViewImage"])
    flat_df.insert(0, "reviewerNote", transaction["reviewerNote"])
    flat_df.insert(0, "savedAt", saved_at)
    flat_df.insert(0, "transactionKey", transaction["transactionKey"])
    flat_df.insert(0, "transactionNo", int(transaction_no))

    return transaction, flat_df[BATCH_COLUMNS], final_total


def invalidate_download_files():
    st.session_state.download_json = None
    st.session_state.download_excel = None
    st.session_state.download_total = 0
    st.session_state.download_transaction_count = 0


def append_current_transaction(reviewer_note):
    if st.session_state.result_df.empty:
        raise ValueError("ยังไม่มีผลวิเคราะห์สำหรับเพิ่มเป็น Transaction")

    transaction, flat_df, final_total = build_transaction(
        df=st.session_state.result_df,
        reviewer_note=reviewer_note,
        ai_result=st.session_state.ai_result,
        transaction_no=st.session_state.transaction_no,
    )

    st.session_state.batch_transactions.append(transaction)
    st.session_state.batch_df = pd.concat(
        [st.session_state.batch_df, flat_df],
        ignore_index=True,
    )
    invalidate_download_files()

    return transaction["transactionNo"], final_total


def clear_current_transaction(advance_transaction=False, clear_download=True):
    st.session_state.ai_result = None
    st.session_state.result_df = pd.DataFrame(columns=CURRENT_COLUMNS)
    st.session_state.raw_json = ""
    st.session_state.widget_version += 1

    if advance_transaction:
        st.session_state.transaction_no += 1

    if clear_download:
        invalidate_download_files()


def reset_all_transactions():
    st.session_state.batch_transactions = []
    st.session_state.batch_df = pd.DataFrame(columns=BATCH_COLUMNS)
    st.session_state.transaction_no = 1
    st.session_state.flash_message = ""
    clear_current_transaction(advance_transaction=False, clear_download=True)


def create_excel_bytes(result_df, transaction_summary_df):
    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Pallet Result")
        transaction_summary_df.to_excel(
            writer,
            index=False,
            sheet_name="Transaction Summary",
        )

    output.seek(0)
    return output.getvalue()


def create_batch_exports():
    transactions = st.session_state.batch_transactions

    if not transactions:
        raise ValueError("ยังไม่มี Transaction สำหรับดาวน์โหลด")

    result_df = st.session_state.batch_df.copy()
    result_df["palletCount"] = pd.to_numeric(
        result_df["palletCount"],
        errors="coerce",
    ).fillna(0).astype(int)

    transaction_summary_df = pd.DataFrame(
        [
            {
                "transactionNo": tx["transactionNo"],
                "transactionKey": tx.get("transactionKey", f"Transaction #{tx['transactionNo']}"),
                "savedAt": tx["savedAt"],
                "reviewerNote": tx["reviewerNote"],
                "finalTotalPallets": tx["finalTotalPallets"],
                "confidence": tx["confidence"],
                "riskOfError": tx["riskOfError"],
                "sideViewImage": tx["sideViewImage"],
                "rearViewImage": tx["rearViewImage"],
            }
            for tx in transactions
        ]
    )

    grand_total = int(result_df["palletCount"].sum())
    payload = {
        "reviewedByUser": True,
        "exportedAt": now_bangkok(),
        "transactionCount": len(transactions),
        "grandTotalPallets": grand_total,
        "transactions": transactions,
    }

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    excel_bytes = create_excel_bytes(result_df, transaction_summary_df)

    return json_text, excel_bytes, grand_total, len(transactions)


def render_image_input(title_th, title_en, view_key, transaction_no, widget_version):
    st.markdown(
        f'<div class="upload-label-box">📷 {title_th}<br>{title_en}</div>',
        unsafe_allow_html=True,
    )

    source = st.radio(
        f"วิธีเลือกรูป {title_th}",
        options=["📁 อัปโหลดรูป", "📷 ถ่ายรูป"],
        horizontal=True,
        key=f"{view_key}_source_{widget_version}",
        label_visibility="collapsed",
    )

    if source == "📷 ถ่ายรูป":
        image_file = st.camera_input(
            f"ถ่ายรูป{title_th}",
            key=f"{view_key}_camera_{widget_version}",
            help="บนมือถือ ระบบจะขอสิทธิ์เปิดกล้องเพื่อถ่ายรูปโดยตรง",
        )
    else:
        image_file = st.file_uploader(
            f"อัปโหลดรูป{title_th}",
            type=["jpg", "jpeg", "png"],
            key=f"{view_key}_upload_{widget_version}",
            label_visibility="collapsed",
        )

    if image_file:
        st.image(
            image_file,
            caption=f"Transaction {transaction_no} — {title_en}",
            use_container_width=True,
        )

    return image_file


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def pair_folder_images(uploaded_files):
    """Pair <base>.1.<ext> as side view with <base>.2.<ext> as rear view."""
    grouped = {}
    ignored = []
    errors = []

    for upload in uploaded_files or []:
        normalized_name = str(getattr(upload, "name", "")).replace("\\", "/")
        path = PurePosixPath(normalized_name)
        file_name = path.name
        extension = path.suffix.lower()
        stem = file_name[:-len(extension)] if extension else file_name
        match = re.match(r"^(?P<base>.+)\.(?P<view>[12])$", stem)

        if not match:
            ignored.append(normalized_name)
            continue

        base_name = match.group("base")
        view_no = match.group("view")
        parent = "" if str(path.parent) == "." else str(path.parent)
        transaction_key = f"{parent}/{base_name}" if parent else base_name
        transaction_group = grouped.setdefault(transaction_key, {})

        if view_no in transaction_group:
            errors.append(
                f"พบไฟล์ซ้ำสำหรับ {transaction_key}.{view_no}: "
                f"{transaction_group[view_no].name} และ {normalized_name}"
            )
            continue

        transaction_group[view_no] = upload

    pairs = []
    for transaction_key in sorted(grouped, key=natural_sort_key):
        views = grouped[transaction_key]
        missing = [view_no for view_no in ("1", "2") if view_no not in views]

        if missing:
            missing_label = ", ".join(f".{view_no}" for view_no in missing)
            errors.append(f"{transaction_key} ขาดไฟล์มุม {missing_label}")
            continue

        pairs.append(
            {
                "folderTransaction": transaction_key,
                "side": views["1"],
                "rear": views["2"],
            }
        )

    return pairs, ignored, errors


def build_folder_pair_preview(pairs):
    return pd.DataFrame(
        [
            {
                "Transaction": pair["folderTransaction"],
                "Side View (.1)": pair["side"].name,
                "Rear View (.2)": pair["rear"].name,
                "Status": "พร้อมวิเคราะห์",
            }
            for pair in pairs
        ]
    )


def analyze_folder_pairs(pairs, hint):
    result_frames = []
    ai_results = {}
    analysis_errors = []
    progress = st.progress(0, text="กำลังเตรียมวิเคราะห์รูปภาพ...")
    total_pairs = len(pairs)

    for index, pair in enumerate(pairs, start=1):
        transaction_key = pair["folderTransaction"]
        progress.progress(
            (index - 1) / total_pairs,
            text=f"กำลังวิเคราะห์ {transaction_key} ({index}/{total_pairs})",
        )

        try:
            result, result_df, _ = analyze_pallets_with_gemini(
                side_upload=pair["side"],
                rear_upload=pair["rear"],
                hint=hint,
                transaction_no=index,
            )

            summary = result.get("summary", {})
            folder_df = result_df.copy()
            folder_df.insert(0, "riskOfError", summary.get("riskOfError", "-"))
            folder_df.insert(0, "confidence", result.get("confidence", "-"))
            folder_df.insert(0, "rearViewImage", pair["rear"].name)
            folder_df.insert(0, "sideViewImage", pair["side"].name)
            folder_df.insert(0, "folderTransaction", transaction_key)
            result_frames.append(folder_df[FOLDER_RESULT_COLUMNS])
            ai_results[transaction_key] = result

        except Exception as exc:
            analysis_errors.append(
                f"{transaction_key}: {type(exc).__name__}: {exc}"
            )

        progress.progress(
            index / total_pairs,
            text=f"วิเคราะห์แล้ว {index}/{total_pairs} คู่",
        )

    progress.empty()

    if result_frames:
        combined_df = pd.concat(result_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=FOLDER_RESULT_COLUMNS)

    return combined_df, ai_results, analysis_errors


def clear_folder_results():
    st.session_state.folder_result_df = pd.DataFrame(columns=FOLDER_RESULT_COLUMNS)
    st.session_state.folder_ai_results = {}
    st.session_state.folder_analysis_errors = []
    st.session_state.folder_widget_version += 1


def append_folder_transactions(folder_df, reviewer_note):
    if folder_df.empty:
        raise ValueError("ยังไม่มีผลวิเคราะห์จากโฟลเดอร์สำหรับบันทึก")

    required_columns = {"folderTransaction", *CURRENT_COLUMNS}
    missing_columns = required_columns.difference(folder_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    transaction_keys = list(dict.fromkeys(folder_df["folderTransaction"].astype(str)))
    start_transaction_no = int(st.session_state.transaction_no)
    new_transactions = []
    new_frames = []
    added_total = 0

    for offset, transaction_key in enumerate(transaction_keys):
        transaction_no = start_transaction_no + offset
        transaction_rows = folder_df.loc[
            folder_df["folderTransaction"].astype(str) == transaction_key,
            CURRENT_COLUMNS,
        ].copy()
        ai_result = st.session_state.folder_ai_results.get(transaction_key, {})

        transaction, flat_df, final_total = build_transaction(
            df=transaction_rows,
            reviewer_note=reviewer_note,
            ai_result=ai_result,
            transaction_no=transaction_no,
            transaction_key=transaction_key,
        )
        transaction["source"] = "folder_upload"
        new_transactions.append(transaction)
        new_frames.append(flat_df)
        added_total += final_total

    st.session_state.batch_transactions.extend(new_transactions)
    st.session_state.batch_df = pd.concat(
        [st.session_state.batch_df, *new_frames],
        ignore_index=True,
    )
    st.session_state.transaction_no = start_transaction_no + len(new_transactions)
    invalidate_download_files()

    return len(new_transactions), added_total


# =========================================================
# TOP BAR
# =========================================================
st.markdown(
    """
<div class="topbar">
    <div class="topbar-logo">📦 PalletVision</div>
    <div class="topbar-badge">AI Counting</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# STEP BAR
# =========================================================
st.markdown(
    """
<div class="stepbar">
    <span class="step-done">1</span>
    <span class="step-active">เลือกรูป</span>
    <div class="step-sep"></div>
    <span class="step-circle-active">2</span>
    <span class="step-active">AI วิเคราะห์</span>
    <div class="step-sep"></div>
    <span class="step-circle">3</span>
    <span>เพิ่ม Transaction</span>
    <div class="step-sep"></div>
    <span class="step-circle">4</span>
    <span>Download รวม</span>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.flash_message:
    st.success(st.session_state.flash_message)
    st.session_state.flash_message = ""

def render_single_transaction_tab():
    st.markdown(f"### Transaction #{st.session_state.transaction_no}")
    st.caption("เลือกอัปโหลดรูปจากเครื่อง หรือถ่ายรูปใหม่จากกล้องมือถือได้ทั้ง 2 มุม")

    widget_version = st.session_state.widget_version
    current_transaction_no = st.session_state.transaction_no
    col1, col2 = st.columns(2)

    with col1:
        side_upload = render_image_input(
            title_th="ด้านข้างรถบรรทุก",
            title_en="Side View",
            view_key="side",
            transaction_no=current_transaction_no,
            widget_version=widget_version,
        )

    with col2:
        rear_upload = render_image_input(
            title_th="ด้านหลังรถบรรทุก",
            title_en="Rear View",
            view_key="rear",
            transaction_no=current_transaction_no,
            widget_version=widget_version,
        )

    hint = st.text_area(
        "Optional Hint",
        value=(
            "Use 3D pallet counting formula: Total Pallets = Height Layers × Width Columns × Depth Rows. "
            "Rear view identifies Width Columns. Side view identifies Depth Rows. "
            "Both views confirm Height Layers."
        ),
        height=90,
        key=f"hint_{widget_version}",
    )

    if st.button(
        "✨ Analyze with AI",
        use_container_width=True,
        key=f"analyze_{widget_version}",
    ):
        if not side_upload or not rear_upload:
            st.markdown(
                '<div class="error-box">กรุณาเลือกรูปทั้ง 2 มุม: ด้านข้าง และด้านหลัง โดยอัปโหลดหรือถ่ายรูปจากกล้อง</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("AI กำลังวิเคราะห์จำนวนพาเลท..."):
                try:
                    result, result_df, raw_json = analyze_pallets_with_gemini(
                        side_upload=side_upload,
                        rear_upload=rear_upload,
                        hint=hint,
                        transaction_no=current_transaction_no,
                    )
                    st.session_state.ai_result = result
                    st.session_state.result_df = result_df
                    st.session_state.raw_json = raw_json
                    invalidate_download_files()
                    st.success("AI วิเคราะห์เสร็จแล้ว")
                except Exception as exc:
                    st.markdown(
                        f'<div class="error-box">AI analysis error: {type(exc).__name__}: {exc}</div>',
                        unsafe_allow_html=True,
                    )

    current_df = st.session_state.result_df.copy()
    result = st.session_state.ai_result or {}

    if current_df.empty:
        st.info("ยังไม่มีผลวิเคราะห์สำหรับ Transaction ปัจจุบัน")
        return

    total = int(pd.to_numeric(current_df["palletCount"], errors="coerce").fillna(0).sum())
    explanation = result.get("countingExplanation", {})
    summary = result.get("summary", {})
    confidence = result.get("confidence", "-")
    risk = summary.get("riskOfError", "-")
    height_layers = safe_int(explanation.get("heightLayers", 0))
    width_columns = safe_int(explanation.get("widthColumns", 0))
    depth_rows = safe_int(explanation.get("depthRows", 0))

    st.markdown(
        f"""
<div class="result-header">
    <div>
        <div class="result-title">✨ ผล AI วิเคราะห์ — Transaction #{current_transaction_no}</div>
        <div class="result-sub">Confidence: {confidence} · Risk of Error: {risk}</div>
    </div>
    <div>
        <div class="result-num">{total}</div>
        <div class="result-num-label">พาเลทรวม</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="info-card">
    <div style="font-size:13px;color:#888;margin-bottom:8px;">
        Counting Formula: Total Pallets = Height Layers × Width Columns × Depth Rows
    </div>
    <div class="info-grid">
        <div class="metric-box"><div class="metric-label">Height Layers</div><div class="metric-value">{height_layers}</div></div>
        <div class="metric-box"><div class="metric-label">Width Columns</div><div class="metric-value">{width_columns}</div></div>
        <div class="metric-box"><div class="metric-label">Depth Rows</div><div class="metric-value">{depth_rows}</div></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="total-bar">
    <span class="total-label">รวม Transaction ปัจจุบัน หลัง AI วิเคราะห์ / หลังแก้ไข</span>
    <span class="total-num">{total} พาเลท</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="note-box">ℹ️ แก้ไขจำนวนหรือสีพาเลทได้ จากนั้นกด “เพิ่มข้อมูล” เพื่อเปิด Transaction ถัดไป</div>',
        unsafe_allow_html=True,
    )

    edited_df = st.data_editor(
        current_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "imageFileName": st.column_config.TextColumn("Image File Name", disabled=False),
            "palletColor": st.column_config.SelectboxColumn(
                "Pallet Color",
                options=["red", "blue", "white", "wood", "black", "green", "other", "unknown"],
                required=True,
            ),
            "palletType": st.column_config.SelectboxColumn(
                "Pallet Type",
                options=PALLET_TYPE_OPTIONS,
                help=(
                    "ค่าเริ่มต้นมาจากสีที่ AI ตรวจจับ: red → Loscam, "
                    "blue → Chep, สีอื่น → Other และ User สามารถแก้ไขได้"
                ),
                required=True,
            ),
            "palletCount": st.column_config.NumberColumn(
                "Pallet Count", min_value=0, max_value=999, step=1, required=True
            ),
        },
        key=f"editable_result_table_{widget_version}",
    )
    st.session_state.result_df = ensure_pallet_type(edited_df)[CURRENT_COLUMNS]

    reviewer_note = st.text_area(
        "Reviewer Note สำหรับ Transaction ปัจจุบัน",
        placeholder="Add your note before adding this transaction...",
        height=80,
        key=f"reviewer_note_{widget_version}",
    )

    with st.expander("ดูรายละเอียดการคำนวณของ AI"):
        st.write("Height Layers Explanation:", explanation.get("heightLayersExplanation", "-"))
        st.write("Width Columns Explanation:", explanation.get("widthColumnsExplanation", "-"))
        st.write("Depth Rows Explanation:", explanation.get("depthRowsExplanation", "-"))
        st.write("Formula:", explanation.get("formula", "-"))
        st.write("Calculation Method:", explanation.get("calculationMethod", "-"))
        st.write("Assumptions:", summary.get("assumptions", "-"))

    action1, action2 = st.columns(2)
    with action1:
        if st.button("🔄 ล้างรอบนี้", use_container_width=True, key="clear_single"):
            clear_current_transaction(advance_transaction=False, clear_download=True)
            st.session_state.flash_message = "ล้างข้อมูล Transaction ปัจจุบันแล้ว"
            st.rerun()

    with action2:
        if st.button("➕ เพิ่มข้อมูล", use_container_width=True, key="add_single"):
            try:
                transaction_no, transaction_total = append_current_transaction(reviewer_note)
                clear_current_transaction(advance_transaction=True, clear_download=False)
                st.session_state.flash_message = (
                    f"เพิ่ม Transaction #{transaction_no} สำเร็จ — {transaction_total} พาเลท"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Add transaction error: {type(exc).__name__}: {exc}")


def render_folder_batch_tab():
    folder_version = st.session_state.folder_widget_version
    st.markdown("### อัปโหลดรูปทั้งโฟลเดอร์")
    st.caption(
        "ตั้งชื่อไฟล์เป็นชื่อ Transaction เดียวกัน โดย .1 = Side View และ .2 = Rear View "
        "เช่น TRX001.1.jpg และ TRX001.2.jpg"
    )

    folder_files = st.file_uploader(
        "เลือกโฟลเดอร์รูปภาพ",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files="directory",
        key=f"folder_upload_{folder_version}",
        help="ระบบจะอัปโหลดไฟล์รูปทั้งหมดในโฟลเดอร์และโฟลเดอร์ย่อย",
    )

    pairs, ignored_files, pairing_errors = pair_folder_images(folder_files)

    if folder_files:
        summary1, summary2, summary3 = st.columns(3)
        summary1.metric("ไฟล์ทั้งหมด", len(folder_files))
        summary2.metric("คู่ที่พร้อมวิเคราะห์", len(pairs))
        summary3.metric("ไฟล์ไม่ตรงรูปแบบ", len(ignored_files))

        if pairs:
            st.dataframe(build_folder_pair_preview(pairs), use_container_width=True, hide_index=True)

        if pairing_errors:
            with st.expander(f"⚠️ ปัญหาการจับคู่ไฟล์ ({len(pairing_errors)})", expanded=True):
                for message in pairing_errors:
                    st.write(f"- {message}")

        if ignored_files:
            with st.expander(f"ไฟล์ที่ไม่ใช้ เพราะชื่อไม่ลงท้าย .1 หรือ .2 ({len(ignored_files)})"):
                for file_name in ignored_files:
                    st.write(f"- {file_name}")

    folder_hint = st.text_area(
        "Optional Hint สำหรับรูปทั้งหมด",
        value=(
            "Use 3D pallet counting formula: Total Pallets = Height Layers × Width Columns × Depth Rows. "
            "File ending .1 is the side view. File ending .2 is the rear view."
        ),
        height=80,
        key=f"folder_hint_{folder_version}",
    )

    if st.button(
        "✨ วิเคราะห์ทั้งโฟลเดอร์",
        use_container_width=True,
        key=f"analyze_folder_{folder_version}",
    ):
        if not folder_files:
            st.error("กรุณาเลือกโฟลเดอร์รูปภาพก่อน")
        elif not pairs:
            st.error("ไม่พบคู่ไฟล์ที่พร้อมวิเคราะห์ กรุณาตรวจชื่อไฟล์ .1 และ .2")
        else:
            result_df, ai_results, analysis_errors = analyze_folder_pairs(pairs, folder_hint)
            st.session_state.folder_result_df = result_df
            st.session_state.folder_ai_results = ai_results
            st.session_state.folder_analysis_errors = analysis_errors
            invalidate_download_files()

            if result_df.empty:
                st.error("AI ไม่สามารถวิเคราะห์คู่รูปใดได้สำเร็จ")
            else:
                st.success(
                    f"AI วิเคราะห์สำเร็จ {result_df['folderTransaction'].nunique()} Transactions"
                )

    if st.session_state.folder_analysis_errors:
        with st.expander(
            f"รายการที่วิเคราะห์ไม่สำเร็จ ({len(st.session_state.folder_analysis_errors)})",
            expanded=True,
        ):
            for message in st.session_state.folder_analysis_errors:
                st.write(f"- {message}")

    folder_df = st.session_state.folder_result_df.copy()
    if folder_df.empty:
        st.info("ผลจากการวิเคราะห์ทั้งโฟลเดอร์จะแสดงเป็นตารางบริเวณนี้")
        return

    st.markdown("### ตารางผล AI สำหรับแก้ไขก่อนบันทึก")
    folder_count = folder_df["folderTransaction"].nunique()
    folder_total = int(pd.to_numeric(folder_df["palletCount"], errors="coerce").fillna(0).sum())
    metric1, metric2 = st.columns(2)
    metric1.metric("จำนวน Transaction", folder_count)
    metric2.metric("จำนวนพาเลทรวม", folder_total)

    edited_folder_df = st.data_editor(
        folder_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        disabled=[
            "folderTransaction",
            "sideViewImage",
            "rearViewImage",
            "confidence",
            "riskOfError",
            "imageFileName",
        ],
        column_config={
            "folderTransaction": st.column_config.TextColumn("Transaction"),
            "sideViewImage": st.column_config.TextColumn("Side View (.1)"),
            "rearViewImage": st.column_config.TextColumn("Rear View (.2)"),
            "confidence": st.column_config.TextColumn("Confidence"),
            "riskOfError": st.column_config.TextColumn("Risk"),
            "imageFileName": st.column_config.TextColumn("Image Pair"),
            "palletColor": st.column_config.SelectboxColumn(
                "Pallet Color",
                options=["red", "blue", "white", "wood", "black", "green", "other", "unknown"],
                required=True,
            ),
            "palletType": st.column_config.SelectboxColumn(
                "Pallet Type",
                options=PALLET_TYPE_OPTIONS,
                help=(
                    "ค่าเริ่มต้นมาจากสีที่ AI ตรวจจับ: red → Loscam, "
                    "blue → Chep, สีอื่น → Other และ User สามารถแก้ไขได้"
                ),
                required=True,
            ),
            "palletCount": st.column_config.NumberColumn(
                "Pallet Count",
                min_value=0,
                max_value=999,
                step=1,
                required=True,
            ),
        },
        key=f"folder_result_editor_{folder_version}",
    )
    st.session_state.folder_result_df = ensure_pallet_type(
        edited_folder_df
    )[FOLDER_RESULT_COLUMNS]

    folder_note = st.text_area(
        "Reviewer Note สำหรับ Batch นี้",
        placeholder="หมายเหตุจะถูกบันทึกให้ทุก Transaction ในโฟลเดอร์นี้",
        height=80,
        key=f"folder_reviewer_note_{folder_version}",
    )

    save1, save2 = st.columns([1, 2])
    with save1:
        if st.button("🔄 ล้างผล Batch", use_container_width=True, key="clear_folder_batch"):
            clear_folder_results()
            st.session_state.flash_message = "ล้างผลวิเคราะห์แบบโฟลเดอร์แล้ว"
            st.rerun()

    with save2:
        if st.button(
            "💾 บันทึกข้อมูลทั้งหมดในตาราง",
            use_container_width=True,
            key="save_folder_batch",
        ):
            try:
                transaction_count, total_pallets = append_folder_transactions(
                    st.session_state.folder_result_df,
                    folder_note,
                )
                clear_folder_results()
                st.session_state.flash_message = (
                    f"บันทึกจากโฟลเดอร์สำเร็จ {transaction_count} Transactions — "
                    f"รวม {total_pallets} พาเลท"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Folder save error: {type(exc).__name__}: {exc}")


def render_saved_transactions_and_download():
    st.markdown("---")
    st.subheader("รายการ Transaction ที่บันทึกแล้ว")

    if st.session_state.batch_transactions:
        batch_total = int(
            pd.to_numeric(st.session_state.batch_df["palletCount"], errors="coerce")
            .fillna(0)
            .sum()
        )
        metric1, metric2 = st.columns(2)
        metric1.metric("จำนวน Transaction", len(st.session_state.batch_transactions))
        metric2.metric("จำนวนพาเลทรวม", batch_total)
        st.dataframe(st.session_state.batch_df, use_container_width=True, hide_index=True)

        action1, action2 = st.columns([2, 1])
        with action1:
            if st.button("💾 เตรียมไฟล์ Download", use_container_width=True, key="prepare_download"):
                try:
                    json_text, excel_bytes, grand_total, transaction_count = create_batch_exports()
                    st.session_state.download_json = json_text.encode("utf-8")
                    st.session_state.download_excel = excel_bytes
                    st.session_state.download_total = grand_total
                    st.session_state.download_transaction_count = transaction_count
                    st.success(
                        f"เตรียมไฟล์สำเร็จ — {transaction_count} Transactions, รวม {grand_total} พาเลท"
                    )
                except Exception as exc:
                    st.error(f"Save error: {type(exc).__name__}: {exc}")

        with action2:
            if st.button("🗑️ ล้างทั้งหมด", use_container_width=True, key="reset_all"):
                reset_all_transactions()
                clear_folder_results()
                st.session_state.flash_message = "ล้างรายการ Transaction ทั้งหมดแล้ว"
                st.rerun()
    else:
        st.caption("ยังไม่มี Transaction ที่บันทึกไว้")

    if st.session_state.download_json is not None:
        st.markdown(
            f"""
<div class="toast-success">
    ✅ พร้อมดาวน์โหลด {st.session_state.download_transaction_count} Transactions
    — รวม {st.session_state.download_total} พาเลท
</div>
""",
            unsafe_allow_html=True,
        )
        download_timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S")
        download1, download2 = st.columns(2)
        with download1:
            st.download_button(
                label="Download JSON Result",
                data=st.session_state.download_json,
                file_name=f"pallet_transactions_{download_timestamp}.json",
                mime="application/json",
                use_container_width=True,
            )
        with download2:
            st.download_button(
                label="Download Excel Result",
                data=st.session_state.download_excel,
                file_name=f"pallet_transactions_{download_timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


# =========================================================
# LEFT SIDEBAR MENU
# =========================================================
with st.sidebar:
    st.markdown("## 📦 PalletVision")
    st.caption("เลือกวิธีนำเข้ารูปภาพ")

    selected_menu = st.radio(
        "Upload Menu",
        options=["Upload by Transaction", "Upload by Folder"],
        format_func=lambda value: {
            "Upload by Transaction": "📷 Upload by Transaction",
            "Upload by Folder": "📁 Upload by Folder",
        }[value],
        key="left_upload_menu",
        label_visibility="collapsed",
    )

    st.markdown("---")
    if selected_menu == "Upload by Transaction":
        st.info("อัปโหลดหรือถ่ายรูป Side View และ Rear View ทีละ Transaction")
    else:
        st.info("อัปโหลดทั้ง Folder โดยใช้ .1 เป็น Side View และ .2 เป็น Rear View")


if selected_menu == "Upload by Transaction":
    render_single_transaction_tab()
else:
    render_folder_batch_tab()

render_saved_transactions_and_download()