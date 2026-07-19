"""
PalletVision — AI Pallet Counting App
Streamlit Gradient Blue Design + Gemini AI Counting

Run:
pip install streamlit pandas pillow openpyxl google-genai
streamlit run pallet_vision_app.py
"""

import json
from datetime import datetime
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
    initial_sidebar_state="collapsed",
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
CURRENT_COLUMNS = ["imageFileName", "palletColor", "palletCount"]
BATCH_COLUMNS = [
    "transactionNo",
    "savedAt",
    "reviewerNote",
    "sideViewImage",
    "rearViewImage",
    "confidence",
    "riskOfError",
    "imageFileName",
    "palletColor",
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
    cleaned_df["palletCount"] = pd.to_numeric(
        cleaned_df["palletCount"],
        errors="coerce",
    ).fillna(0).astype(int)

    return cleaned_df[CURRENT_COLUMNS]


def build_transaction(df, reviewer_note, ai_result, transaction_no):
    cleaned_df = validate_result_df(df)
    saved_at = now_bangkok()
    result = ai_result or {}
    summary = result.get("summary", {})
    file_names = result.get("fileName", {})
    final_total = int(cleaned_df["palletCount"].sum())

    transaction = {
        "transactionNo": int(transaction_no),
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

st.markdown(f"### Transaction #{st.session_state.transaction_no}")
st.caption("เลือกอัปโหลดรูปจากเครื่อง หรือถ่ายรูปใหม่จากกล้องมือถือได้ทั้ง 2 มุม")


# =========================================================
# UPLOAD / CAMERA SECTION
# =========================================================
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


# =========================================================
# ANALYZE BUTTON
# =========================================================
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
                result, df, raw_json = analyze_pallets_with_gemini(
                    side_upload=side_upload,
                    rear_upload=rear_upload,
                    hint=hint,
                    transaction_no=current_transaction_no,
                )

                st.session_state.ai_result = result
                st.session_state.result_df = df
                st.session_state.raw_json = raw_json
                invalidate_download_files()
                st.success("AI วิเคราะห์เสร็จแล้ว")

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">AI analysis error: {type(e).__name__}: {e}</div>',
                    unsafe_allow_html=True,
                )


# =========================================================
# CURRENT RESULT SECTION
# =========================================================
df = st.session_state.result_df.copy()
result = st.session_state.ai_result or {}

if not df.empty:
    if "palletCount" in df.columns:
        total = int(pd.to_numeric(df["palletCount"], errors="coerce").fillna(0).sum())
    else:
        total = 0

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
        <div class="metric-box">
            <div class="metric-label">Height Layers</div>
            <div class="metric-value">{height_layers}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Width Columns</div>
            <div class="metric-value">{width_columns}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Depth Rows</div>
            <div class="metric-value">{depth_rows}</div>
        </div>
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
        """
<div class="note-box">
    ℹ️ แก้ไขจำนวนหรือสีพาเลทได้ จากนั้นกด “เพิ่มข้อมูล” เพื่อเปิด Transaction ถัดไป
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Current Transaction Result")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "imageFileName": st.column_config.TextColumn(
                "Image File Name",
                disabled=False,
            ),
            "palletColor": st.column_config.SelectboxColumn(
                "Pallet Color",
                options=[
                    "red",
                    "blue",
                    "white",
                    "wood",
                    "black",
                    "green",
                    "other",
                    "unknown",
                ],
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
        key=f"editable_result_table_{widget_version}",
    )

    st.session_state.result_df = edited_df

    with st.expander("ดูรายละเอียดการคำนวณของ AI"):
        st.write("Height Layers Explanation:")
        st.write(explanation.get("heightLayersExplanation", "-"))

        st.write("Width Columns Explanation:")
        st.write(explanation.get("widthColumnsExplanation", "-"))

        st.write("Depth Rows Explanation:")
        st.write(explanation.get("depthRowsExplanation", "-"))

        st.write("Formula:")
        st.write(explanation.get("formula", "-"))

        st.write("Calculation Method:")
        st.write(explanation.get("calculationMethod", "-"))

        st.write("Assumptions:")
        st.write(summary.get("assumptions", "-"))

    with st.expander("Raw AI JSON"):
        st.code(
            json.dumps(result, indent=2, ensure_ascii=False),
            language="json",
        )
else:
    st.info("ยังไม่มีผลวิเคราะห์สำหรับ Transaction ปัจจุบัน")


# =========================================================
# REVIEWER NOTE
# =========================================================
reviewer_note = st.text_area(
    "Reviewer Note สำหรับ Transaction ปัจจุบัน",
    placeholder="Add your note before adding this transaction...",
    height=90,
    key=f"reviewer_note_{widget_version}",
)


# =========================================================
# ACCUMULATED TRANSACTIONS
# =========================================================
st.markdown("---")
st.subheader("รายการ Transaction ที่เพิ่มแล้ว")

if st.session_state.batch_transactions:
    batch_total = int(
        pd.to_numeric(
            st.session_state.batch_df["palletCount"],
            errors="coerce",
        ).fillna(0).sum()
    )

    m1, m2 = st.columns(2)
    m1.metric("จำนวน Transaction", len(st.session_state.batch_transactions))
    m2.metric("จำนวนพาเลทรวม", batch_total)

    st.dataframe(
        st.session_state.batch_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("ยังไม่มี Transaction ที่เพิ่มไว้")


# =========================================================
# ACTION BUTTONS
# =========================================================
st.markdown("---")
a1, a2, a3 = st.columns([1, 1.35, 1.65])

with a1:
    if st.button("🔄 ล้างรอบนี้", use_container_width=True):
        clear_current_transaction(advance_transaction=False, clear_download=True)
        st.session_state.flash_message = "ล้างข้อมูล Transaction ปัจจุบันแล้ว"
        st.rerun()

with a2:
    if st.button("➕ เพิ่มข้อมูล", use_container_width=True):
        try:
            transaction_no, transaction_total = append_current_transaction(reviewer_note)
            clear_current_transaction(advance_transaction=True, clear_download=False)
            st.session_state.flash_message = (
                f"เพิ่ม Transaction #{transaction_no} สำเร็จ — {transaction_total} พาเลท "
                "กรุณาเพิ่มรูปสำหรับ Transaction ถัดไป"
            )
            st.rerun()
        except Exception as e:
            st.markdown(
                f'<div class="error-box">Add transaction error: {type(e).__name__}: {e}</div>',
                unsafe_allow_html=True,
            )

with a3:
    if st.button("💾 ยืนยันและเตรียม Download", use_container_width=True):
        try:
            # Save the current analyzed transaction automatically if the user has
            # not pressed “เพิ่มข้อมูล” yet.
            if not st.session_state.result_df.empty:
                append_current_transaction(reviewer_note)
                clear_current_transaction(advance_transaction=True, clear_download=False)

            json_text, excel_bytes, grand_total, transaction_count = create_batch_exports()

            st.session_state.download_json = json_text.encode("utf-8")
            st.session_state.download_excel = excel_bytes
            st.session_state.download_total = grand_total
            st.session_state.download_transaction_count = transaction_count

            st.success(
                f"เตรียมไฟล์สำเร็จ — {transaction_count} Transactions, "
                f"รวม {grand_total} พาเลท"
            )

        except Exception as e:
            st.markdown(
                f'<div class="error-box">Save error: {type(e).__name__}: {e}</div>',
                unsafe_allow_html=True,
            )


# =========================================================
# DOWNLOAD SECTION
# =========================================================
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
    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download JSON Result",
            data=st.session_state.download_json,
            file_name=f"pallet_transactions_{download_timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            label="Download Excel Result",
            data=st.session_state.download_excel,
            file_name=f"pallet_transactions_{download_timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

if st.session_state.batch_transactions:
    if st.button("🗑️ ล้าง Transaction ทั้งหมด", use_container_width=True):
        reset_all_transactions()
        st.session_state.flash_message = "ล้างรายการ Transaction ทั้งหมดแล้ว"
        st.rerun()