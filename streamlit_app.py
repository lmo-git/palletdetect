"""
PalletVision — AI Pallet Counting App
Streamlit Gradient Blue Design + Gemini AI Counting

Run:
pip install streamlit pandas pillow openpyxl google-genai
streamlit run pallet_vision_app.py
"""

import json
from datetime import datetime

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
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "result_df" not in st.session_state:
    st.session_state.result_df = pd.DataFrame(
        columns=["imageFileName", "palletColor", "palletCount"]
    )

if "saved" not in st.session_state:
    st.session_state.saved = False

if "raw_json" not in st.session_state:
    st.session_state.raw_json = ""


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


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
        df = pd.DataFrame(columns=["imageFileName", "palletColor", "palletCount"])

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
        errors="coerce"
    ).fillna(0).astype(int)

    return df[["imageFileName", "palletColor", "palletCount"]]


def analyze_pallets_with_gemini(side_upload, rear_upload, hint):
    side_file_name = side_upload.name
    rear_file_name = rear_upload.name
    combined_file_name = f"{side_file_name} + {rear_file_name}"

    side_img = Image.open(side_upload)
    rear_img = Image.open(rear_upload)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PALLET_PROMPT,
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

    df = normalize_result_rows(result, combined_file_name)

    return result, df, raw_text


def save_reviewed_result(df, reviewer_note):
    df = pd.DataFrame(df)

    required_cols = ["imageFileName", "palletColor", "palletCount"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["palletCount"] = pd.to_numeric(
        df["palletCount"],
        errors="coerce"
    ).fillna(0).astype(int)

    final_total = int(df["palletCount"].sum())

    final_result = {
        "reviewedByUser": True,
        "savedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finalTotalPallets": final_total,
        "reviewerNote": reviewer_note or "",
        "resultRows": df.to_dict(orient="records"),
        "aiRawResult": st.session_state.ai_result,
    }

    json_text = json.dumps(final_result, indent=2, ensure_ascii=False)
    excel_bytes = create_excel_bytes(df)

    return json_text, excel_bytes, final_total


def create_excel_bytes(df):
    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Pallet Result")

    output.seek(0)
    return output.getvalue()


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
    <span class="step-active">เลือกรูปภาพ</span>
    <div class="step-sep"></div>
    <span class="step-circle-active">2</span>
    <span class="step-active">AI วิเคราะห์</span>
    <div class="step-sep"></div>
    <span class="step-circle">3</span>
    <span>ยืนยันและบันทึก</span>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD SECTION
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="upload-label-box">📷 ด้านข้างรถบรรทุก<br>Side View</div>',
        unsafe_allow_html=True,
    )

    side_upload = st.file_uploader(
        "Upload side view",
        type=["jpg", "jpeg", "png"],
        key="side_upload",
        label_visibility="collapsed",
    )

    if side_upload:
        st.image(side_upload, use_container_width=True)

with col2:
    st.markdown(
        '<div class="upload-label-box">📷 ด้านหลังรถบรรทุก<br>Rear View</div>',
        unsafe_allow_html=True,
    )

    rear_upload = st.file_uploader(
        "Upload rear view",
        type=["jpg", "jpeg", "png"],
        key="rear_upload",
        label_visibility="collapsed",
    )

    if rear_upload:
        st.image(rear_upload, use_container_width=True)


hint = st.text_area(
    "Optional Hint",
    value=(
        "Use 3D pallet counting formula: Total Pallets = Height Layers × Width Columns × Depth Rows. "
        "Rear view identifies Width Columns. Side view identifies Depth Rows. "
        "Both views confirm Height Layers."
    ),
    height=90,
)


# =========================================================
# ANALYZE BUTTON
# =========================================================
if st.button("✨ Analyze with AI", use_container_width=True):
    if not side_upload or not rear_upload:
        st.markdown(
            '<div class="error-box">กรุณาอัปโหลดรูปภาพทั้ง 2 มุม: ด้านข้าง และ ด้านหลัง</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("AI กำลังวิเคราะห์จำนวนพาเลท..."):
            try:
                result, df, raw_json = analyze_pallets_with_gemini(
                    side_upload=side_upload,
                    rear_upload=rear_upload,
                    hint=hint,
                )

                st.session_state.ai_result = result
                st.session_state.result_df = df
                st.session_state.raw_json = raw_json
                st.session_state.saved = False

                st.success("AI วิเคราะห์เสร็จแล้ว")

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">AI analysis error: {type(e).__name__}: {e}</div>',
                    unsafe_allow_html=True,
                )


# =========================================================
# RESULT SECTION
# =========================================================
df = st.session_state.result_df.copy()

if "palletCount" in df.columns:
    total = int(pd.to_numeric(df["palletCount"], errors="coerce").fillna(0).sum())
else:
    total = 0

result = st.session_state.ai_result or {}
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
        <div class="result-title">✨ ผล AI วิเคราะห์</div>
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
    <span class="total-label">รวมทั้งหมด หลัง AI วิเคราะห์ / หลังแก้ไข</span>
    <span class="total-num">{total} พาเลท</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="note-box">
    ℹ️ User สามารถแก้ไขจำนวนพาเลทหรือสีพาเลทได้ก่อนกดยืนยันและบันทึกข้อมูล
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# EDITABLE TABLE
# =========================================================
st.subheader("Result Table")

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
    key="editable_result_table",
)

st.session_state.result_df = edited_df


# =========================================================
# COUNTING EXPLANATION
# =========================================================
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
        json.dumps(result, indent=2, ensure_ascii=False) if result else "{}",
        language="json",
    )


# =========================================================
# SAVE SECTION
# =========================================================
reviewer_note = st.text_area(
    "Reviewer Note",
    placeholder="Add your note before saving...",
    height=90,
)

b1, b2 = st.columns([1, 2])

with b1:
    if st.button("🔄 เริ่มใหม่", use_container_width=True):
        st.session_state.ai_result = None
        st.session_state.result_df = pd.DataFrame(
            columns=["imageFileName", "palletColor", "palletCount"]
        )
        st.session_state.saved = False
        st.session_state.raw_json = ""

        if "editable_result_table" in st.session_state:
            del st.session_state["editable_result_table"]

        st.rerun()

with b2:
    label = "✅ บันทึกแล้ว" if st.session_state.saved else "💾 ยืนยันและบันทึกข้อมูล"

    if st.button(label, use_container_width=True):
        if st.session_state.result_df.empty:
            st.markdown(
                '<div class="error-box">ยังไม่มีข้อมูลสำหรับบันทึก กรุณากด Analyze with AI ก่อน</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                json_text, excel_bytes, final_total = save_reviewed_result(
                    st.session_state.result_df,
                    reviewer_note,
                )

                st.session_state.saved = True

                st.markdown(
                    f"""
                <div class="toast-success">
                    ✅ บันทึกข้อมูลสำเร็จ — รวม {final_total} พาเลท
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.download_button(
                    label="Download JSON Result",
                    data=json_text.encode("utf-8"),
                    file_name="pallet_final_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

                st.download_button(
                    label="Download Excel Result",
                    data=excel_bytes,
                    file_name="pallet_final_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">Save error: {type(e).__name__}: {e}</div>',
                    unsafe_allow_html=True,
                )