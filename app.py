import streamlit as st
import pandas as pd
import joblib

# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="Bike Buyer Prediction",
    page_icon="🏍️",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("BikeBuyerModel.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# CSS
# ==========================

st.markdown("""
<style>

/* Ẩn menu */
#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Background */

.stApp{

background:
linear-gradient(135deg,#020617,#111827);

color:white;

}

/* Main */

.block-container{

max-width:1250px;

padding-top:35px;

}

/* Tiêu đề */

.title{

font-size:48px;

font-weight:900;

color:white;

letter-spacing:1px;

}

.subtitle{

font-size:19px;

color:#94a3b8;

margin-top:10px;

margin-bottom:20px;

}

.badge{

display:inline-block;

padding:14px 28px;

border-radius:40px;

background:linear-gradient(90deg,#2563eb,#9333ea);

color:white;

font-size:18px;

font-weight:700;

}

/* Card */

.card{

background:#0f172a;

border:1px solid #334155;

border-radius:25px;

padding:30px;

box-shadow:0 20px 45px rgba(0,0,0,.35);

}

/* Streamlit container */

div[data-testid="stVerticalBlock"]{

background:transparent;

}

div[data-testid="stHorizontalBlock"]{

background:transparent;

}

/* Label */

label{

color:#cbd5e1 !important;

font-weight:700;

}

/* Input */

.stNumberInput input{

background:#1e293b !important;

color:white !important;

}

.stSelectbox div[data-baseweb="select"]>div{

background:#1e293b;

color:white;

border-radius:12px;

}

/* Slider */

.stSlider{

padding-top:15px;

padding-bottom:20px;

}

/* Button */

.stButton>button{

width:100%;

height:58px;

border:none;

border-radius:15px;

background:linear-gradient(90deg,#2563eb,#7c3aed);

color:white;

font-size:20px;

font-weight:bold;

transition:.25s;

}

.stButton>button:hover{

transform:translateY(-2px);

box-shadow:0 8px 20px rgba(59,130,246,.5);

}

/* Metric */

div[data-testid="metric-container"]{

background:#1e293b;

padding:18px;

border-radius:15px;

border:1px solid #334155;

}

/* Result */

.result-success{

padding:35px;

border-radius:20px;

background:#052e16;

border:2px solid #22c55e;

text-align:center;

font-size:36px;

font-weight:bold;

color:#22c55e;

}

.result-fail{

padding:35px;

border-radius:20px;

background:#450a0a;

border:2px solid #ef4444;

text-align:center;

font-size:36px;

font-weight:bold;

color:#ef4444;

}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================

st.markdown("""

<div class="title">
 HỆ THỐNG DỰ ĐOÁN KHẢ NĂNG MUA XE MÁY
</div>

<div class="subtitle">
Ứng dụng Machine Learning sử dụng thuật toán Random Forest Classification
</div>

<div class="badge">
 Accuracy : 93.62%
</div>

""", unsafe_allow_html=True)

st.write("")

# ==========================
# LAYOUT
# ==========================

left, right = st.columns([1,1], gap="large")
# =====================================================
# INPUT
# =====================================================

with left:

    st.markdown("""
    <div style="background:#0f172a;padding:30px;border-radius:25px;
    border:1px solid #334155;box-shadow:0 20px 45px rgba(0,0,0,.35);">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## THÔNG TIN KHÁCH HÀNG")

    income = st.number_input(
        "Thu nhập",
        min_value=0,
        value=50000,
        step=1000,
        format="%d"
    )

    age = st.slider(
        "Tuổi",
        18,
        100,
        35
    )

    col1, col2 = st.columns(2)

    with col1:
        children = st.number_input(
            "Số con",
            0,
            10,
            1
        )

    with col2:
        cars = st.number_input(
            "Số xe đang có",
            0,
            10,
            1
        )

    gender = st.selectbox(
        "Giới tính",
        [
            "Nam",
            "Nữ"
        ]
    )

    marital = st.selectbox(
        "Tình trạng hôn nhân",
        [
            "Đã kết hôn",
            "Độc thân"
        ]
    )

    education = st.selectbox(
        "Trình độ học vấn",
        [
            "Đại học",
            "Sau đại học",
            "Trung học",
            "Cao đẳng",
            "Chưa tốt nghiệp THPT"
        ]
    )

    occupation = st.selectbox(
        "Nghề nghiệp",
        [
            "Nhân viên",
            "Quản lý",
            "Lao động",
            "Chuyên gia",
            "Kỹ thuật"
        ]
    )

    home = st.selectbox(
        "Sở hữu nhà",
        [
            "Có",
            "Không"
        ]
    )

    commute = st.selectbox(
        "Khoảng cách đi làm",
        [
            "0-1 Miles",
            "1-2 Miles",
            "2-5 Miles",
            "5-10 Miles",
            "10+ Miles"
        ]
    )

    region = st.selectbox(
        "Khu vực",
        [
            "Châu Âu",
            "Bắc Mỹ",
            "Thái Bình Dương"
        ]
    )

    st.write("")

    predict = st.button(
        "DỰ ĐOÁN"
    )
    # =====================================================
# RESULT
# =====================================================

with right:

    st.subheader("KẾT QUẢ DỰ ĐOÁN")

    if predict:

        # Chuyển đổi dữ liệu
        gender_value = "Male" if gender == "Nam" else "Female"
        marital_value = "Single" if marital == "Độc thân" else "Married"

        data = pd.DataFrame({

            "ID":[0],
            "Income":[income],
            "Children":[children],
            "Cars":[cars],
            "Age":[age],

            "Marital Status_Single":[1 if marital_value=="Single" else 0],

            "Gender_Male":[1 if gender_value=="Male" else 0],

            "Education_Graduate Degree":[1 if education=="Sau đại học" else 0],
            "Education_High School":[1 if education=="Trung học" else 0],
            "Education_Partial College":[1 if education=="Cao đẳng" else 0],
            "Education_Partial High School":[1 if education=="Chưa tốt nghiệp THPT" else 0],

            "Occupation_Management":[1 if occupation=="Quản lý" else 0],
            "Occupation_Manual":[1 if occupation=="Lao động" else 0],
            "Occupation_Professional":[1 if occupation=="Chuyên gia" else 0],
            "Occupation_Skilled Manual":[1 if occupation=="Kỹ thuật" else 0],

            "Home Owner_Yes":[1 if home=="Có" else 0],

            "Commute Distance_1-2 Miles":[1 if commute=="1-2 Miles" else 0],
            "Commute Distance_10+ Miles":[1 if commute=="10+ Miles" else 0],
            "Commute Distance_2-5 Miles":[1 if commute=="2-5 Miles" else 0],
            "Commute Distance_5-10 Miles":[1 if commute=="5-10 Miles" else 0],

            "Region_North America":[1 if region=="Bắc Mỹ" else 0],
            "Region_Pacific":[1 if region=="Thái Bình Dương" else 0]

        })

        # Chuẩn hóa
        X = scaler.transform(data)

        # Dự đoán
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        if prediction == 1:

            percent = probability[1]

            st.success("KHÁCH HÀNG CÓ KHẢ NĂNG MUA XE")

        else:

            percent = probability[0]

            st.error("KHÁCH HÀNG KHÔNG CÓ KHẢ NĂNG MUA XE")

        st.write("")

        st.metric(
            label="Xác suất dự đoán",
            value=f"{percent*100:.2f}%"
        )

        st.progress(float(percent))

        st.write("")

        st.subheader("Thông tin đầu vào")

        result = pd.DataFrame({

            "Thông tin":[
                "Thu nhập",
                "Tuổi",
                "Số con",
                "Số xe",
                "Giới tính",
                "Hôn nhân",
                "Học vấn",
                "Nghề nghiệp",
                "Sở hữu nhà",
                "Khoảng cách đi làm",
                "Khu vực"
            ],

            "Giá trị":[
                f"{income:,}",
                age,
                children,
                cars,
                gender,
                marital,
                education,
                occupation,
                home,
                commute,
                region
            ]

        })

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("Nhập thông tin khách hàng rồi nhấn DỰ ĐOÁN.")
