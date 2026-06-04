import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
from io import BytesIO
from datetime import datetime
from pathlib import Path
import os


st.set_page_config(page_title="Pallet Detection", layout="wide")


# =========================
# PATH
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pallet_rnn_count.pt"


# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: #f8fafc;
}

h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #f8fafc !important;
}

input, textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
}

[data-baseweb="select"] > div {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #444 !important;
}

[data-baseweb="select"] span {
    color: #ffffff !important;
}

div[role="listbox"] {
    background-color: #000000 !important;
    border: 1px solid #333 !important;
}

div[role="option"] {
    background-color: #000000 !important;
    color: #ffffff !important;
}

div[role="option"]:hover {
    background-color: #1f2937 !important;
    color: #38f8a6 !important;
}

[data-testid="stFileUploader"] section {
    border: 2px solid #000000 !important;
    border-radius: 14px !important;
    background: #f8fafc !important;
}

[data-testid="stFileUploader"] section * {
    color: #000000 !important;
}

[data-testid="stFileUploader"] button {
    color: #000000 !important;
    border: 1px solid #000000 !important;
    background: #ffffff !important;
}

.main-card {
    background: linear-gradient(180deg, #111d33 0%, #0a1222 100%);
    border-radius: 26px;
    padding: 24px;
    border: 1px solid #2b456d;
    box-shadow: 0 18px 50px rgba(0,0,0,0.45);
}

.metric-card {
    background: #16243d;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #2b456d;
    text-align: center;
}

.big-number {
    font-size: 46px;
    font-weight: 800;
    color: #38f8a6 !important;
}

.label {
    color: #b7c7e6 !important;
    font-size: 14px;
    letter-spacing: 1px;
}

.stButton > button {
    background: #22c55e;
    color: white !important;
    border-radius: 16px;
    height: 52px;
    border: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# =========================
# MODEL CLASS
# =========================
class PalletRNNCounter(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.rnn = nn.GRU(
            input_size=64 * 28,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, h, c * w)

        out, _ = self.rnn(x)

        return self.fc(out[:, -1, :])


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"ไม่พบไฟล์ model: {MODEL_PATH}")
        st.write("Current Directory:", os.getcwd())
        st.write("BASE_DIR:", BASE_DIR)
        st.write("Files in BASE_DIR:", [f.name for f in BASE_DIR.iterdir()])
        st.stop()

    model = PalletRNNCounter()

    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error("โหลด model ไม่สำเร็จ")
        st.exception(e)
        st.stop()

    model.eval()
    return model


model = load_model()


# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================
# SESSION
# =========================
if "records" not in st.session_state:
    st.session_state.records = []


# =========================
# FUNCTIONS
# =========================
def predict_count(image):
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor).item()

    return pred, max(0, round(pred))


def draw_dashed_line(draw, start, end, fill, width=4, dash_length=22, gap_length=12):
    x1, y1 = start
    x2, y2 = end

    total_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    if total_len == 0:
        return

    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len

    dist = 0

    while dist < total_len:
        dash_end = min(dist + dash_length, total_len)

        sx = x1 + dx * dist
        sy = y1 + dy * dist
        ex = x1 + dx * dash_end
        ey = y1 + dy * dash_end

        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

        dist += dash_length + gap_length


def draw_measurement_overlay(image, count):
    img = image.copy().convert("RGB")

    blue_overlay = Image.new("RGB", img.size, (0, 130, 200))
    img = Image.blend(img, blue_overlay, 0.10)
    img = img.filter(ImageFilter.SHARPEN)

    draw = ImageDraw.Draw(img)

    w, h = img.size

    if count <= 0:
        return img

    x1 = int(w * 0.18)
    x2 = int(w * 0.82)

    top_margin = int(h * 0.14)
    bottom_margin = int(h * 0.88)

    area_height = bottom_margin - top_margin
    step = area_height / count

    yellow = (255, 220, 60)

    for i in range(count):
        y = int(top_margin + step * i + step / 2)

        draw_dashed_line(
            draw=draw,
            start=(x1, y),
            end=(x2, y),
            fill=yellow,
            width=4
        )

        draw.text(
            (x1 - 55, y - 12),
            f"#{i + 1}",
            fill=yellow
        )

    return img


def to_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Pallet Count"
        )

    return output.getvalue()


# =========================
# UI
# =========================
st.title("Pallet Detection")

st.caption(
    "Upload/ถ่ายรูป → RNN Predict → ปรับความแม่นยำ → แสดงเส้นวัดสีเหลือง → Confirm → Export Excel"
)

with st.expander("Debug Model Path", expanded=False):
    st.write("Current Directory:", os.getcwd())
    st.write("BASE_DIR:", str(BASE_DIR))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("MODEL EXISTS:", MODEL_PATH.exists())
    st.write("MODEL SIZE MB:", round(MODEL_PATH.stat().st_size / 1024 / 1024, 2) if MODEL_PATH.exists() else None)


with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    top1, top2 = st.columns([1, 1])

    with top1:
        weight_ticket = st.text_input("เลขตั๋วชั่ง")

    with top2:
        image_source = st.radio(
            "เลือกวิธีนำเข้ารูป",
            ["Upload รูปภาพ", "ถ่ายรูปจากกล้อง"],
            horizontal=True,
            index=0
        )

    uploaded_image = None

    if image_source == "Upload รูปภาพ":
        uploaded_image = st.file_uploader(
            "Upload รูปพาเลท",
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_image = st.camera_input("ถ่ายรูปพาเลท")

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

        raw_pred, base_count = predict_count(image)

        img_col, side_col = st.columns([2.2, 1])

        with side_col:
            st.markdown("### Accuracy")

            accuracy_slider = st.slider(
                "ปรับความแม่นยำ",
                min_value=0.50,
                max_value=1.50,
                value=1.00,
                step=0.01
            )

            adjusted_count = max(0, round(raw_pred * accuracy_slider))

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="big-number">{adjusted_count}</div>
                    <div class="label">PALLET FOUND</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric("Raw Prediction", f"{raw_pred:.2f}")
            st.metric("Model Count", base_count)

            pallet_type = st.selectbox(
                "Confirm ประเภทพาเลท",
                [
                    "Red Pallet",
                    "CHEP",
                    "Plastic Pallet",
                    "Wooden Pallet",
                    "Unknown"
                ]
            )

            confirmed_count = st.number_input(
                "Confirm จำนวนพาเลท",
                min_value=0,
                value=int(adjusted_count),
                step=1
            )

            remark = st.text_area("Remark")

        highlighted_img = draw_measurement_overlay(
            image=image,
            count=int(adjusted_count)
        )

        with img_col:
            st.image(
                highlighted_img,
                caption="Measurement Overlay",
                use_container_width=True
            )

        if st.button("บันทึกผลตรวจ", type="primary"):
            if weight_ticket.strip() == "":
                st.error("กรุณากรอกเลขตั๋วชั่ง")
            else:
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "weight_ticket": weight_ticket,
                    "model_path": str(MODEL_PATH),
                    "model_type": "CNN + GRU/RNN Regression",
                    "raw_prediction": round(raw_pred, 3),
                    "model_count": base_count,
                    "accuracy_slider": accuracy_slider,
                    "adjusted_count": adjusted_count,
                    "confirmed_count": confirmed_count,
                    "pallet_type": pallet_type,
                    "remark": remark,
                }

                st.session_state.records.append(record)
                st.success("บันทึกข้อมูลเรียบร้อย")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# HISTORY
# =========================
st.markdown("## History")

if len(st.session_state.records) > 0:
    df = pd.DataFrame(st.session_state.records)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download Excel",
        data=to_excel(df),
        file_name="pallet_detection_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("ยังไม่มีข้อมูลที่บันทึก")