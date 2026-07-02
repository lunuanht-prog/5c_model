import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# KHỐI MASTER: CẤU HÌNH TRANG VÀ HÀM CHUNG
# ==========================================
st.set_page_config(layout="wide", page_title="Dự báo Phân loại bằng Logistic Regression", page_icon="🤖")

# Định nghĩa hằng số từ notebook
TARGET = 'PD'
FEATURES = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 'NL4', 
            'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5', 
            'V6', 'TS1', 'TS2', 'TS3', 'TS4']

@st.cache_data
def load_data(file):
    """Hàm nạp dữ liệu dùng chung"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# Khởi tạo session state
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu (CSV/Excel)", type=["csv", "xlsx", "xls"])
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Logistic Regression")
    
    # Cấu hình siêu tham số
    test_size = st.slider("Tỷ lệ tập kiểm thử (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Tỷ lệ chia dữ liệu cho tập kiểm thử. Mặc định trong notebook là 0.2")
    random_state = st.number_input("Hạt giống ngẫu nhiên (random_state)", min_value=0, value=23, step=1, help="Đảm bảo kết quả chia tập train/test ổn định qua các lần chạy.")
    
    with st.expander("Tham số nâng cao (Logistic Regression)"):
        c_param = st.number_input("Hệ số C (Nghịch đảo của chuẩn hóa)", min_value=0.01, value=1.0, step=0.1)
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=2000, value=100, step=100)
    
    st.divider()
    train_btn = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🤖 Ứng Dụng Dự Báo Mô Hình Phân Loại")
st.caption("Ứng dụng xây dựng trên thuật toán Logistic Regression nhằm dự báo biến mục tiêu (PD) dựa trên nhóm đặc trưng (TC, NL, DK, V, TS).")

if not uploaded_file:
    st.info("👈 Vui lòng tải lên tệp dữ liệu mẫu ở thanh bên trái (Sidebar) để bắt đầu.")
    st.stop()

# Nạp dữ liệu
try:
    df = load_data(uploaded_file)
    st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
except Exception as e:
    st.error(f"Lỗi khi đọc file: {e}")
    st.stop()

# Kiểm tra dữ liệu hợp lệ
missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_cols:
    st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
    st.stop()

st.divider()

# ==========================================
# KHỐI XỬ LÝ HUẤN LUYỆN (Kích hoạt từ Sidebar)
# ==========================================
if train_btn:
    with st.spinner("Đang huấn luyện mô hình..."):
        # Chuẩn bị dữ liệu
        X = df[FEATURES]
        y = df[TARGET]
        
        # Split train/test (theo thông số từ notebook: test_size=0.2, random_state=23)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Khởi tạo và huấn luyện
        model = LogisticRegression(C=c_param, max_iter=max_iter)
        model.fit(X_train, y_train)
        
        # Dự báo và đánh giá
        y_pred = model.predict(X_test)
        
        # Lưu vào session state
        st.session_state.model = model
        st.session_state.is_trained = True
        st.session_state.acc = accuracy_score(y_test, y_pred)
        st.session_state.report = classification_report(y_test, y_pred, output_dict=True)
        st.session_state.cm = confusion_matrix(y_test, y_pred)
        
        st.success("🎉 Đã huấn luyện mô hình thành công! Vui lòng xem kết quả ở các tab bên dưới.")

# ==========================================
# GIAO DIỆN CHÍNH: TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan dữ liệu", "📈 Trực quan hóa", "🎯 Kết quả mô hình", "🔮 Sử dụng mô hình"])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    st.subheader("Kích thước dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số dòng", df.shape[0])
    col2.metric("Tổng số cột", df.shape[1])
    col3.metric("Dung lượng file (KB)", round(uploaded_file.size / 1024, 2))
    
    st.subheader("Dữ liệu thô (5 dòng đầu)")
    with st.container(height=250):
        st.dataframe(df.head(), use_container_width=True)
        
    st.subheader("Thống kê mô tả các biến đặc trưng (X) và biến mục tiêu (y)")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("Phân phối của các biến")
    
    selected_vis_features = st.multiselect(
        "Chọn biến để trực quan hóa (Tối đa 4 biến để hiển thị đẹp nhất):", 
        options=FEATURES, 
        default=FEATURES[:3] # Mặc định chọn 3 biến đầu
    )
    
    # Lưới 2x2
    cols = st.columns(2)
    
    # 1. Vẽ biến mục tiêu trước (Categorical/Binary)
    with cols[0]:
        fig_target = px.histogram(df, x=TARGET, title=f"Phân phối biến mục tiêu ({TARGET})", color=TARGET, text_auto=True)
        fig_target.update_layout(xaxis_type='category')
        st.plotly_chart(fig_target, use_container_width=True)
        
    # 2. Vẽ các biến đầu vào được chọn
    for i, feature in enumerate(selected_vis_features):
        col_idx = (i + 1) % 2
        with cols[col_idx]:
            # Giả định các biến này là dạng số (Likert scale 1-5 theo như data mẫu)
            fig = px.histogram(df, x=feature, title=f"Phân phối biến {feature}", nbins=10)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ---
with tab3:
    if not st.session_state.is_trained:
        st.info("ℹ️ Vui lòng bấm nút **Huấn luyện mô hình** ở thanh bên trái để xem kết quả kiểm định.")
    else:
        st.subheader("Chỉ tiêu đánh giá (Tập kiểm thử)")
        
        report = st.session_state.report
        
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Độ chính xác (Accuracy)", f"{st.session_state.acc:.2%}")
        met2.metric("Precision (Macro Avg)", f"{report['macro avg']['precision']:.2%}")
        met3.metric("Recall (Macro Avg)", f"{report['macro avg']['recall']:.2%}")
        met4.metric("F1-Score (Macro Avg)", f"{report['macro avg']['f1-score']:.2%}")
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
            cm = st.session_state.cm
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Nhãn Dự Báo", y="Nhãn Thực Tế", color="Số lượng"),
                               x=[str(c) for c in range(cm.shape[1])],
                               y=[str(c) for c in range(cm.shape[0])])
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col2:
            st.subheader("Báo cáo phân loại (Classification Report)")
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH ---
with tab4:
    if not st.session_state.is_trained:
        st.info("ℹ️ Vui lòng huấn luyện mô hình trước khi sử dụng để dự báo dữ liệu mới.")
    else:
        mode = st.radio("Chọn phương thức dự báo:", ["Nhập tay từng biến", "Tải lên tệp CSV/Excel"])
        
        if mode == "Nhập tay từng biến":
            st.subheader("Nhập giá trị đặc trưng")
            with st.form("predict_form"):
                # Tạo lưới 4 cột để nhập form cho gọn
                input_cols = st.columns(4)
                input_data = {}
                
                for i, feature in enumerate(FEATURES):
                    # Lấy min/max và trung vị từ tập dữ liệu hiện tại làm mặc định
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                    med_val = int(df[feature].median())
                    
                    with input_cols[i % 4]:
                        input_data[feature] = st.number_input(label=feature, min_value=min_val, max_value=max_val, value=med_val)
                
                submitted = st.form_submit_button("🔮 Dự báo", type="primary")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    prediction = st.session_state.model.predict(input_df)[0]
                    prob = st.session_state.model.predict_proba(input_df)[0]
                    
                    st.success(f"**Kết quả dự báo (PD): {prediction}**")
                    st.info(f"Xác suất: Lớp 0 = {prob[0]:.1%}, Lớp 1 = {prob[1]:.1%}")
                    
        else:
            st.subheader("Dự báo hàng loạt")
            upload_test = st.file_uploader("Tải lên dữ liệu cần dự báo (Chỉ cần các cột biến X)", type=["csv", "xlsx"])
            
            if upload_test:
                try:
                    df_test = load_data(upload_test)
                    test_missing = [col for col in FEATURES if col not in df_test.columns]
                    
                    if test_missing:
                        st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {', '.join(test_missing)}")
                    else:
                        X_new = df_test[FEATURES]
                        preds = st.session_state.model.predict(X_new)
                        
                        df_result = df_test.copy()
                        df_result['Prediction_PD'] = preds
                        
                        st.success("Dự báo hoàn tất!")
                        with st.container(height=300):
                            st.dataframe(df_result, use_container_width=True)
                            
                        # Xuất CSV
                        csv_data = df_result.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải kết quả (CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Đã có lỗi xảy ra khi xử lý file: {e}")
