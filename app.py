import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import base64
from io import BytesIO
import sqlite3
import threading
from contextlib import contextmanager

# Import các module tùy chỉnh
from pinns_optimizer import optimize_dam_section, create_force_diagram, plot_loss_curve
from database import DamDatabase

# Khởi tạo cơ sở dữ liệu
@st.cache_resource(ttl=3600)
def get_database():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'dam_results.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = DamDatabase(db_path)
    db.create_tables()
    return db

# Hàm tạo báo cáo Excel
def create_excel_report(result):
    """
    Tạo báo cáo Excel từ kết quả tính toán
    
    Parameters:
    -----------
    result : dict
        Kết quả tính toán từ hàm optimize_dam_section
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu báo cáo
    """
    # Tạo DataFrame cho báo cáo
    data = {
        'Thông số': [
            'Chiều cao đập (H)',
            'Trọng lượng riêng bê tông (γ_bt)',
            'Trọng lượng riêng nước (γ_n)',
            'Hệ số ma sát (f)',
            'Cường độ kháng cắt (C)',
            'Hệ số ổn định yêu cầu (Kc)',
            'Hệ số áp lực thấm (a1)',
            'Hệ số mái thượng lưu (n)',
            'Hệ số mái hạ lưu (m)',
            'Tham số ξ',
            'Diện tích mặt cắt (A)',
            'Hệ số ổn định (K)',
            'Ứng suất mép thượng lưu (σ)',
            'Chiều rộng đáy đập (B)',
            'Trọng lượng bản thân đập (G)',
            'Trọng lượng phần dốc (G1)',
            'Trọng lượng phần đứng (G2)',
            'Áp lực nước thượng lưu - tam giác (W1)',
            'Áp lực nước thượng lưu - hình chữ nhật (W2_1)',
            'Áp lực nước thượng lưu - tam giác (W2_2)',
            'Tổng áp lực nước thượng lưu (W2)',
            'Áp lực thấm (Wt)',
            'Lực chống trượt (Fct)',
            'Lực gây trượt (Fgt)',
            'Độ lệch tâm (e)',
            'Số vòng lặp thực tế',
            'Số vòng lặp tối đa',
            'Thời gian tính toán'
        ],
        'Giá trị': [
            f"{result['H']:.2f} m",
            f"{result['gamma_bt']:.2f} T/m³",
            f"{result['gamma_n']:.2f} T/m³",
            f"{result['f']:.2f}",
            f"{result['C']:.2f} T/m²",
            f"{result['Kc']:.2f}",
            f"{result['a1']:.2f}",
            f"{result['n']:.4f}",
            f"{result['m']:.4f}",
            f"{result['xi']:.4f}",
            f"{result['A']:.2f} m²",
            f"{result['K']:.4f}",
            f"{result['sigma']:.4f} T/m²",
            f"{result['B']:.2f} m",
            f"{result['G']:.2f} T",
            f"{result['G1']:.2f} T",
            f"{result['G2']:.2f} T",
            f"{result['W1']:.2f} T",
            f"{result['W2_1']:.2f} T",
            f"{result['W2_2']:.2f} T",
            f"{result['W2']:.2f} T",
            f"{result['Wt']:.2f} T",
            f"{result['Fct']:.2f} T",
            f"{result['Fgt']:.2f} T",
            f"{result['e']:.2f} m",
            f"{result['iterations']}",
            f"{result.get('max_iterations', 5000)}",
            f"{result.get('computation_time', 0):.2f} giây"
        ]
    }
    
    df = pd.DataFrame(data)
    return df

# Hàm tạo link tải xuống Excel
def get_excel_download_link(df, filename="bao_cao.xlsx"):
    """
    Tạo link tải xuống file Excel
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu báo cáo
    filename : str
        Tên file Excel
        
    Returns:
    --------
    href : str
        Link tải xuống
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Báo cáo')
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Tải xuống báo cáo Excel</a>'
    return href

# Thiết lập ứng dụng Streamlit
st.set_page_config(
    page_title="Tính toán tối ưu mặt cắt đập bê tông",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Khởi tạo session state mặc định nếu chưa có ---
default_values = {
    'H': 60.0,
    'gamma_bt': 2.4,
    'gamma_n': 1.0,
    'f': 0.7,
    'C': 0.5,
    'Kc': 1.2,
    'a1': 0.6,
    'max_iterations': 5000,
    'convergence_threshold': 1e-6,
    'patience': 50,
    'result': None,
    'show_history': False
}
for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# CSS tùy chỉnh theo phong cách Apple
st.markdown("""
<style>
    /* Phông chữ và màu sắc */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        color: #333333;
        background-color: #ffffff;
    }
    
    /* Tiêu đề */
    h1, h2, h3 {
        font-weight: 500;
        color: #000000;
    }
    
    /* Nút */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0055aa;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f5f5f7;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    
    /* Form */
    .stForm {
        background-color: #f5f5f7;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Thẻ thông tin */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Footer */
    .footer {
        position: relative;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        color: #666666;
        margin-top: 50px;
    }
    
    /* Metric */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 500;
        color: #0066cc;
    }
    
    /* Thẻ thành công */
    .stSuccess {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    
    /* Thẻ cảnh báo */
    .stWarning {
        background-color: #fff8e1;
        border-left-color: #ff9800;
    }
    
    /* Thẻ lỗi */
    .stError {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề ứng dụng
st.title("Công cụ tính toán tối ưu mặt cắt đập bê tông trọng lực")
st.markdown("Sử dụng mô hình Physics-Informed Neural Networks (PINNs)")

# Tạo tabs
tabs = st.tabs(["Tính toán", "Lịch sử tính toán", "Lý thuyết", "Giới thiệu"])

# Tab Tính toán
with tabs[0]:
    # Chia layout thành 2 cột
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Thông số đầu vào")
        
        # Form nhập liệu
        with st.form("input_form"):
            H = st.number_input("Chiều cao đập H (m)", min_value=10.0, max_value=300.0, value=st.session_state.H, step=5.0)
            
            st.markdown("#### Thông số vật liệu và nền")
            gamma_bt = st.number_input("Trọng lượng riêng bê tông γ_bt (T/m³)", min_value=2.0, max_value=3.0, value=st.session_state.gamma_bt, step=0.1)
            gamma_n = st.number_input("Trọng lượng riêng nước γ_n (T/m³)", min_value=0.9, max_value=1.1, value=st.session_state.gamma_n, step=0.1)
            f = st.number_input("Hệ số ma sát f", min_value=0.3, max_value=0.9, value=st.session_state.f, step=0.05)
            C = st.number_input("Cường độ kháng cắt C (T/m²)", min_value=0.0, max_value=10.0, value=st.session_state.C, step=0.1)
            
            st.markdown("#### Thông số ổn định và thấm")
            Kc = st.number_input("Hệ số ổn định yêu cầu Kc", min_value=1.0, max_value=2.0, value=st.session_state.Kc, step=0.1)
            a1 = st.number_input("Hệ số áp lực thấm α1", min_value=0.0, max_value=1.0, value=st.session_state.a1, step=0.1)
            
            st.markdown("#### Thông số tính toán PINNs")
            max_iterations = st.slider("Số vòng lặp tối đa", min_value=1000, max_value=10000, value=st.session_state.max_iterations, step=1000)
            
            # Thông số hội tụ nâng cao (có thể ẩn đi)
            show_advanced = st.checkbox("Hiển thị thông số nâng cao")
            if show_advanced:
                convergence_threshold = st.number_input(
                    "Ngưỡng hội tụ", 
                    min_value=1e-8, 
                    max_value=1e-4, 
                    value=st.session_state.convergence_threshold, 
                    format="%.1e"
                )
                patience = st.slider(
                    "Số vòng lặp kiên nhẫn", 
                    min_value=10, 
                    max_value=200, 
                    value=st.session_state.patience, 
                    step=10,
                    help="Số vòng lặp chờ đợi khi không có cải thiện trước khi dừng sớm"
                )
            else:
                convergence_threshold = st.session_state.convergence_threshold
                patience = st.session_state.patience
            
            # Nút tính toán
            submitted = st.form_submit_button("Tính toán tối ưu")
        
        # Nút đặt lại
        reset_clicked = st.button("🔄 Đặt lại")
        if reset_clicked:
            for k, v in default_values.items():
                st.session_state[k] = v
            st.experimental_rerun()
    
        # Xử lý khi form được gửi
        if submitted:
            # Cập nhật session state
            st.session_state.H = H
            st.session_state.gamma_bt = gamma_bt
            st.session_state.gamma_n = gamma_n
            st.session_state.f = f
            st.session_state.C = C
            st.session_state.Kc = Kc
            st.session_state.a1 = a1
            st.session_state.max_iterations = max_iterations
            st.session_state.convergence_threshold = convergence_threshold
            st.session_state.patience = patience
            
            with st.spinner("Đang tính toán tối ưu mặt cắt đập..."):
                # Ghi lại thời gian bắt đầu
                start_time = datetime.now()
                
                # Thực hiện tính toán
                result = optimize_dam_section(
                    H=H,
                    gamma_bt=gamma_bt,
                    gamma_n=gamma_n,
                    f=f,
                    C=C,
                    Kc=Kc,
                    a1=a1,
                    max_iterations=max_iterations,
                    convergence_threshold=convergence_threshold,
                    patience=patience
                )
                
                # Tính thời gian tính toán
                computation_time = (datetime.now() - start_time).total_seconds()
                result['computation_time'] = computation_time
                
                # Lưu kết quả vào session state
                st.session_state['result'] = result
                
                try:
                    # Lưu kết quả vào cơ sở dữ liệu
                    db = get_database()
                    result_id = db.save_result(result)
                    st.session_state['last_result_id'] = result_id
                    st.success(f"Đã lưu kết quả tính toán vào cơ sở dữ liệu (ID: {result_id})")
                except Exception as e:
                    st.warning(f"Không thể lưu kết quả vào cơ sở dữ liệu: {str(e)}")
    
    # Hiển thị kết quả nếu có
    with col2:
        if 'result' in st.session_state and st.session_state['result'] is not None:
            result = st.session_state['result']
            
            st.markdown("### Kết quả tính toán")
            
            # Hiển thị các tham số tối ưu
            col_params1, col_params2 = st.columns(2)
            
            with col_params1:
                st.metric("Hệ số mái thượng lưu (n)", f"{result['n']:.4f}")
                st.metric("Hệ số mái hạ lưu (m)", f"{result['m']:.4f}")
                st.metric("Tham số ξ", f"{result['xi']:.4f}")
            
            with col_params2:
                st.metric("Diện tích mặt cắt (A)", f"{result['A']:.4f} m²")
                st.metric("Hệ số ổn định (K)", f"{result['K']:.4f}")
                st.metric("Ứng suất mép thượng lưu (σ)", f"{result['sigma']:.4f} T/m²")
            
            # Hiển thị trạng thái
            if abs(result['K'] - result['Kc']) < 0.05:  # Sai số cho phép 5%
                st.success(f"Mặt cắt đập thỏa mãn điều kiện ổn định (K = {result['K']:.4f} ≈ Kc = {result['Kc']:.2f})")
            elif result['K'] > result['Kc']:
                st.info(f"Mặt cắt đập thỏa mãn điều kiện ổn định (K = {result['K']:.4f} > Kc = {result['Kc']:.2f})")
            else:
                st.error(f"Mặt cắt đập KHÔNG thỏa mãn điều kiện ổn định (K = {result['K']:.4f} < Kc = {result['Kc']:.2f})")
            
            if result['sigma'] <= 0:
                st.success(f"Mặt cắt đập thỏa mãn điều kiện không kéo (σ = {result['sigma']:.4f} T/m² ≤ 0)")
            else:
                st.warning(f"Mặt cắt đập có ứng suất kéo ở mép thượng lưu (σ = {result['sigma']:.4f} T/m² > 0)")
            
            # Hiển thị thông tin về số vòng lặp
            st.info(f"Số vòng lặp thực tế: {result['iterations']} / {result.get('max_iterations', max_iterations)} (tối đa)")
            
            # Hiển thị thời gian tính toán
            st.info(f"Thời gian tính toán: {result['computation_time']:.2f} giây")
            
            # Tạo tabs cho các biểu đồ
            result_tabs = st.tabs(["Mặt cắt đập", "Biểu đồ hàm mất mát", "Xuất báo cáo"])
            
            # Tab mặt cắt đập
            with result_tabs[0]:
                # Tạo biểu đồ Plotly tương tác
                try:
                    fig = create_force_diagram(result)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Không thể tạo biểu đồ mặt cắt đập: {str(e)}")
            
            # Tab biểu đồ hàm mất mát
            with result_tabs[1]:
                # Tạo biểu đồ Plotly tương tác
                try:
                    if 'loss_history' in result and len(result['loss_history']) > 0:
                        loss_fig = plot_loss_curve(result['loss_history'])
                        st.plotly_chart(loss_fig, use_container_width=True)
                    else:
                        st.warning("Không có dữ liệu lịch sử hàm mất mát để hiển thị")
                except Exception as e:
                    st.error(f"Không thể tạo biểu đồ hàm mất mát: {str(e)}")
            
            # Tab xuất báo cáo
            with result_tabs[2]:
                st.markdown("### Xuất báo cáo")
                
                # Tạo báo cáo Excel
                excel_df = create_excel_report(result)
                
                # Hiển thị báo cáo
                st.dataframe(excel_df, use_container_width=True)
                
                # Tạo link tải xuống
                st.markdown(
                    get_excel_download_link(excel_df, f"bao_cao_dam_H{int(result['H'])}.xlsx"),
                    unsafe_allow_html=True
                )

# Tab Lịch sử tính toán
with tabs[1]:
    st.markdown("### Lịch sử tính toán")
    
    try:
        # Lấy dữ liệu từ cơ sở dữ liệu
        db = get_database()
        history_df = db.get_all_results()
        
        if len(history_df) > 0:
            # Hiển thị bảng lịch sử
            st.dataframe(
                history_df[['id', 'timestamp', 'H', 'gamma_bt', 'gamma_n', 'f', 'C', 'Kc', 'a1', 'n', 'm', 'xi', 'A', 'K', 'sigma']],
                use_container_width=True
            )
            
            # Chọn kết quả để xem chi tiết
            selected_id = st.selectbox("Chọn ID để xem chi tiết:", history_df['id'].tolist())
            
            if st.button("Xem chi tiết"):
                # Lấy kết quả từ cơ sở dữ liệu
                selected_result = db.get_result_by_id(selected_id)
                
                if selected_result:
                    # Hiển thị thông tin chi tiết
                    st.markdown("#### Thông tin chi tiết")
                    
                    # Tạo DataFrame từ kết quả
                    detail_df = pd.DataFrame({
                        'Thông số': [
                            'Chiều cao đập (H)',
                            'Trọng lượng riêng bê tông (γ_bt)',
                            'Trọng lượng riêng nước (γ_n)',
                            'Hệ số ma sát (f)',
                            'Cường độ kháng cắt (C)',
                            'Hệ số ổn định yêu cầu (Kc)',
                            'Hệ số áp lực thấm (a1)',
                            'Hệ số mái thượng lưu (n)',
                            'Hệ số mái hạ lưu (m)',
                            'Tham số ξ',
                            'Diện tích mặt cắt (A)',
                            'Hệ số ổn định (K)',
                            'Ứng suất mép thượng lưu (σ)',
                            'Số vòng lặp thực tế',
                            'Số vòng lặp tối đa',
                            'Thời gian tính toán',
                            'Thời điểm tính toán'
                        ],
                        'Giá trị': [
                            f"{selected_result['H']:.2f} m",
                            f"{selected_result['gamma_bt']:.2f} T/m³",
                            f"{selected_result['gamma_n']:.2f} T/m³",
                            f"{selected_result['f']:.2f}",
                            f"{selected_result['C']:.2f} T/m²",
                            f"{selected_result['Kc']:.2f}",
                            f"{selected_result['a1']:.2f}",
                            f"{selected_result['n']:.4f}",
                            f"{selected_result['m']:.4f}",
                            f"{selected_result['xi']:.4f}",
                            f"{selected_result['A']:.2f} m²",
                            f"{selected_result['K']:.4f}",
                            f"{selected_result['sigma']:.4f} T/m²",
                            f"{selected_result['iterations']}",
                            f"{selected_result.get('max_iterations', 5000)}",
                            f"{selected_result.get('computation_time', 0):.2f} giây",
                            f"{selected_result['timestamp']}"
                        ]
                    })
                    
                    # Hiển thị DataFrame
                    st.dataframe(detail_df, use_container_width=True)
                    
                    # Tạo link tải xuống Excel
                    st.markdown(
                        get_excel_download_link(detail_df, f"bao_cao_dam_id{selected_id}.xlsx"),
                        unsafe_allow_html=True
                    )
                    
                    # Nút để tải kết quả vào form tính toán
                    if st.button("Tải thông số này vào form tính toán"):
                        st.session_state.H = selected_result['H']
                        st.session_state.gamma_bt = selected_result['gamma_bt']
                        st.session_state.gamma_n = selected_result['gamma_n']
                        st.session_state.f = selected_result['f']
                        st.session_state.C = selected_result['C']
                        st.session_state.Kc = selected_result['Kc']
                        st.session_state.a1 = selected_result['a1']
                        st.session_state.max_iterations = selected_result.get('max_iterations', 5000)
                        st.success("Đã tải thông số vào form tính toán. Chuyển sang tab 'Tính toán' để tiếp tục.")
        else:
            st.info("Chưa có kết quả tính toán nào được lưu trong cơ sở dữ liệu.")
    except Exception as e:
        st.error(f"Lỗi khi truy cập cơ sở dữ liệu: {str(e)}")
        st.info("Vui lòng thực hiện tính toán mới để tạo dữ liệu.")

# Tab Lý thuyết
with tabs[2]:
    st.markdown("""
    ### Lý thuyết tính toán mặt cắt đập bê tông trọng lực sử dụng PINNs
    
    #### Physics-Informed Neural Networks (PINNs)
    
    PINNs là một phương pháp kết hợp giữa mạng nơ-ron học sâu và các ràng buộc vật lý. Trong ứng dụng này, PINNs được sử dụng để tìm các tham số tối ưu của mặt cắt đập bê tông trọng lực, đảm bảo các điều kiện ổn định và an toàn, đồng thời tối thiểu hóa diện tích mặt cắt.
    
    #### Các tham số tối ưu
    
    Mặt cắt đập bê tông trọng lực được mô tả bởi ba tham số chính:
    
    - **n**: Hệ số mái thượng lưu
    - **m**: Hệ số mái hạ lưu
    - **ξ (xi)**: Tham số xác định vị trí điểm gãy khúc trên mái thượng lưu
    
    #### Các điều kiện ràng buộc
    
    Mặt cắt đập phải thỏa mãn các điều kiện sau:
    
    1. **Điều kiện ổn định trượt**: Hệ số ổn định K = Kc
    2. **Điều kiện không kéo**: Ứng suất mép thượng lưu σ ≤ 0
    3. **Tối thiểu hóa diện tích mặt cắt**: Giảm thiểu lượng bê tông sử dụng
    
    #### Công thức tính toán
    
    Các công thức chính được sử dụng trong tính toán:
    
    - **Diện tích mặt cắt**: A = 0.5 * H² * (m + n * (1-ξ)²)
    - **Hệ số ổn định**: K = Fct / Fgt
        - Fct = f * (G + W2 - Wt) + C * H * (m + n * (1-ξ))
        - Fgt = 0.5 * γ_n * H²
    - **Ứng suất mép thượng lưu**: σ = P / B - 6 * M0 / B²
    
    #### Quá trình tối ưu hóa
    
    1. Khởi tạo mạng nơ-ron với các tham số ngẫu nhiên
    2. Tính toán các đại lượng vật lý (A, K, σ) dựa trên đầu ra của mạng
    3. Tính toán hàm mất mát dựa trên các ràng buộc vật lý
    4. Cập nhật trọng số mạng theo hướng giảm gradient của hàm mất mát
    5. Lặp lại quá trình cho đến khi hội tụ
    
    #### Hàm mất mát
    
    Hàm mất mát bao gồm các thành phần:
    
    1. Phạt nếu K khác Kc (đảm bảo ổn định chính xác)
    2. Phạt nếu σ > 0 (đảm bảo không kéo)
    3. Tối thiểu hóa diện tích A
    
    #### Cơ chế hội tụ sớm
    
    Ứng dụng sử dụng cơ chế hội tụ sớm để tối ưu hóa quá trình tính toán:
    
    1. **Ngưỡng hội tụ**: Dừng khi sự thay đổi của hàm mất mát nhỏ hơn ngưỡng
    2. **Kiên nhẫn**: Dừng khi không có cải thiện sau một số vòng lặp nhất định
    3. **Điều chỉnh learning rate**: Giảm learning rate khi hàm mất mát không giảm
    
    Cơ chế này giúp giảm thời gian tính toán và tránh overfitting, đồng thời vẫn đảm bảo tìm được giải pháp tối ưu.
    """)

# Tab Giới thiệu
with tabs[3]:
    st.markdown("""
    ### Giới thiệu về ứng dụng
    
    Ứng dụng này sử dụng mô hình Physics-Informed Neural Networks (PINNs) để tính toán mặt cắt kinh tế đập bê tông trọng lực thỏa mãn các điều kiện ổn định và an toàn.
    
    #### Tính năng chính
    
    - **Tối ưu hóa mặt cắt**: Tìm bộ 3 thông số (n, m, ξ) tối ưu thỏa mãn các điều kiện
    - **Trực quan hóa**: Hiển thị sơ đồ mặt cắt đập và biểu đồ hàm mất mát
    - **Lưu trữ kết quả**: Lưu và truy xuất các kết quả tính toán
    - **Xuất báo cáo**: Tạo báo cáo Excel với đầy đủ thông tin
    
    #### Ưu điểm của phương pháp PINNs
    
    - **Tự động tối ưu hóa**: Không cần thử nghiệm thủ công nhiều phương án
    - **Kết hợp vật lý và học máy**: Đảm bảo kết quả thỏa mãn các định luật vật lý
    - **Hội tụ nhanh**: Cơ chế hội tụ sớm giúp giảm thời gian tính toán
    - **Chính xác cao**: Tìm được mặt cắt tối ưu thỏa mãn chính xác các điều kiện
    
    #### Hướng dẫn sử dụng
    
    1. Nhập các thông số đầu vào ở tab "Tính toán"
    2. Nhấn nút "Tính toán tối ưu" để bắt đầu quá trình tối ưu hóa
    3. Xem kết quả và biểu đồ trực quan
    4. Tải xuống báo cáo Excel nếu cần
    5. Xem lịch sử tính toán ở tab "Lịch sử tính toán"
    
    #### Lưu ý
    
    - Quá trình tối ưu hóa có thể mất từ vài giây đến vài phút tùy thuộc vào thông số đầu vào
    - Số vòng lặp tối đa có thể điều chỉnh để cân bằng giữa thời gian tính toán và độ chính xác
    - Cơ chế hội tụ sớm sẽ tự động dừng quá trình khi đã tìm được giải pháp tối ưu
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>Ứng dụng tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs</p>
</div>
""", unsafe_allow_html=True)
