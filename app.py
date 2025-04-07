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
from database import DamDatabase

# Thêm class mạng PINNs
device = "cuda" if torch.cuda.is_available() else "cpu"

class OptimalParamsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        # Giới hạn đầu ra chính xác như trong file gốc
        n = out[:, 0] * 0.4             # n ∈ [0, 0.4]
        m = out[:, 1] * 3.5 + 0.5       # m ∈ [0.5, 4.0]
        xi = out[:, 2] * 0.99 + 0.01    # xi ∈ (0.01, 1]
        return n, m, xi

# Hàm tính vật lý dùng PINNs - chính xác như trong file gốc
def compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1):
    B = H * (m + n * (1 - xi))
    G1 = 0.5 * gamma_bt * m * H**2
    G2 = 0.5 * gamma_bt * n * H**2 * (1 - xi)**2
    G = G1 + G2
    W1 = 0.5 * gamma_n * H**2
    W2_1 = gamma_n * n * (1 - xi) * xi * H**2
    W2_2 = 0.5 * gamma_n * n * H**2 * (1 - xi)**2
    W2 = W2_1 + W2_2
    Wt = 0.5 * gamma_n * a1 * H * (m * H + n * H * (1 - xi))
    P = G + W2 - Wt
    lG1 = H * (m / 6 - n * (1 - xi) / 2)
    lG2 = H * (m / 2 - n * (1 - xi) / 6)
    lt  = H * (m + n * (1 - xi)) / 6
    l2  = H * m / 2
    l22 = H * m / 2 + H * n * (1 - xi) / 6
    l1  = H / 3
    M0 = -G1 * lG1 - G2 * lG2 + Wt * lt - W2_1 * l2 - W2_2 * l22 + W1 * l1
    sigma = P / B - 6 * M0 / B**2
    Fct = f * (G + W2 - Wt) + C * H * (m + n * (1 - xi))
    Fgt = 0.5 * gamma_n * H**2
    K = Fct / Fgt
    A = 0.5 * H**2 * (m + n * (1 - xi)**2)
    return sigma, K, A, G, W1, W2, Wt, Fct, Fgt, B, M0, G1, G2, W2_1, W2_2, lG1, lG2, lt, l2, l22, l1, P

# Hàm mất mát - chính xác như trong file gốc
def loss_function(sigma, K, A, Kc, alpha):
    # Hệ số k_factor mặc định là 1.0
    k_factor = 1.0
    # K_min = Kc*factor
    K_min = Kc * k_factor

    # Tăng penalty rất lớn nếu K < K_min
    BIG_PENALTY = 1e5
    penalty_K = torch.clamp(K_min - K, min=0)**2
    penalty_K = BIG_PENALTY * penalty_K

    # Ở đây vẫn phạt sigma**2, có thể tùy biến
    penalty_sigma = sigma**2

    # Ghép lại
    return penalty_K.mean() + 100 * penalty_sigma.mean() + alpha * A.mean()

# Thay đổi hàm optimize_dam_section để sử dụng PINNs
def optimize_dam_section(H, gamma_bt, gamma_n, f, C, Kc, a1, max_iterations=5000):
    alpha = 0.01  # hệ số phạt diện tích - giống file gốc
    model = OptimalParamsNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data = torch.ones((1, 1), device=device)
    
    # Lưu lịch sử tối ưu hóa
    loss_history = []

    for epoch in range(max_iterations):
        optimizer.zero_grad()
        n, m, xi = model(data)
        sigma, K, A = compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1)[:3]
        loss = loss_function(sigma, K, A, Kc, alpha)
        loss.backward()
        optimizer.step()
        
        # Lưu lịch sử loss
        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    model.eval()
    n, m, xi = model(data)
    sigma, K, A, G, W1, W2, Wt, Fct, Fgt, B, M0, G1, G2, W2_1, W2_2, lG1, lG2, lt, l2, l22, l1, P = compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1)
    
    # Tính độ lệch tâm
    e = B/2 - M0/P

    return {
        'H': H,
        'gamma_bt': gamma_bt,
        'gamma_n': gamma_n,
        'f': f,
        'C': C,
        'Kc': Kc,
        'a1': a1,
        'n': n.item(),
        'm': m.item(),
        'xi': xi.item(),
        'A': A.item(),
        'K': K.item(),
        'sigma': sigma.item(),
        'G': G.item(),
        'G1': G1.item(),
        'G2': G2.item(),
        'W1': W1.item(),
        'W2': W2.item(),
        'W2_1': W2_1.item(),
        'W2_2': W2_2.item(),
        'Wt': Wt.item(),
        'Fct': Fct.item(),
        'Fgt': Fgt.item(),
        'B': B.item(),
        'e': e.item(),
        'M0': M0.item(),
        'P': P.item(),
        'lG1': lG1.item(),
        'lG2': lG2.item(),
        'lt': lt.item(),
        'l2': l2.item(),
        'l22': l22.item(),
        'l1': l1.item(),
        'iterations': max_iterations,
        'loss_history': loss_history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


# Hàm tạo biểu đồ mặt cắt đập và sơ đồ lực - cập nhật theo file gốc
def create_force_diagram(result):
    import plotly.graph_objects as go

    H = result['H']
    n = result['n']
    m = result['m']
    xi = result['xi']

    B = H * (m + n * (1 - xi))
    x0 = 0
    x1 = n * H * (1 - xi)
    x4 = x1 + m * H

    x = [x0, x1, x1, x1, x4, x0]
    y = [0, H * (1 - xi), H, H, 0, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', fill='toself',
        fillcolor='rgba(211,211,211,0.5)',  # lightgrey với alpha=0.5
        line=dict(color='black', width=1.5),
        name='Mặt cắt đập'
    ))

    # Tính vị trí đặt lực - chính xác như trong file gốc
    mid = B / 2
    lG1 = result['lG1']
    lG2 = result['lG2']
    lt = result['lt']
    l2 = result['l2']
    l22 = result['l22']
    l1 = result['l1']
    
    # Độ dài mũi tên
    arrow_len = H / 15

    # G1 – trọng lượng phần dốc (⬇️)
    fig.add_annotation(
        x=mid - lG1, y=H / 3,
        ax=mid - lG1, ay=H / 3 - arrow_len,
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='red', text='G1',
        font=dict(size=12, color='black')
    )

    # G2 – trọng lượng phần đứng (⬇️)
    fig.add_annotation(
        x=mid - lG2, y=H * (1 - xi) / 3,
        ax=mid - lG2, ay=H * (1 - xi) / 3 - arrow_len,
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='red', text='G2',
        font=dict(size=12, color='black')
    )

    # Wt – áp lực thấm (⬆️)
    fig.add_annotation(
        x=mid - lt, y=0,
        ax=mid - lt, ay=arrow_len,
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='red', text='Wt',
        font=dict(size=12, color='black')
    )

    # W'2 – phần hình chữ nhật (⬇️)
    fig.add_annotation(
        x=mid - l2, y=H * (1 - xi) + xi * H / 2,
        ax=mid - l2, ay=H * (1 - xi) + xi * H / 2 - arrow_len,
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='red', text="W'2",
        font=dict(size=12, color='black')
    )

    # W"2 – phần tam giác (⬇️)
    fig.add_annotation(
        x=mid - l22, y=(2 / 3) * H * (1 - xi),
        ax=mid - l22, ay=(2 / 3) * H * (1 - xi) - arrow_len,
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='red', text='W"2',
        font=dict(size=12, color='black')
    )

    # W1 – áp lực tam giác từ thượng lưu (➡️)
    fig.add_annotation(
        x=x0 - arrow_len, y=l1,
        ax=x0, ay=l1,
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='red', text='W1',
        font=dict(size=12, color='black')
    )

    # Cấu hình chung
    fig.update_layout(
        title=f'Sơ đồ lực tác dụng lên đập H = {H:.0f} m',
        xaxis_title='Chiều rộng (m)',
        yaxis_title='Chiều cao (m)',
        width=850,
        height=600,
        plot_bgcolor='white',
        showlegend=False
    )
    
    # Đặt tỷ lệ trục x và y bằng nhau
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # Đặt giới hạn trục để có không gian cho mũi tên
    fig.update_xaxes(range=[-arrow_len*3, B + arrow_len*3])
    fig.update_yaxes(range=[0, H + arrow_len*3])
    
    # Thêm lưới
    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        )
    )

    return fig


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
            'Số vòng lặp',
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
            f"{result['timestamp']}"
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

# Khởi tạo cơ sở dữ liệu
@st.cache_resource
def get_database():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'dam_results.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return DamDatabase(db_path)

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
            
            st.markdown("#### Thông số tính toán")
            max_iterations = st.slider("Số vòng lặp tối đa", min_value=1000, max_value=10000, value=st.session_state.max_iterations, step=1000)
            
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
                    max_iterations=max_iterations
                )
                
                # Tính thời gian tính toán
                computation_time = (datetime.now() - start_time).total_seconds()
                result['computation_time'] = computation_time
                
                # Lưu kết quả vào session state
                st.session_state['result'] = result
                
                # Lưu kết quả vào cơ sở dữ liệu
                db = get_database()
                result_id = db.save_result(result)
                st.session_state['last_result_id'] = result_id
                st.success(f"Đã lưu kết quả tính toán vào cơ sở dữ liệu (ID: {result_id})")
    
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
            if result['K'] >= result['Kc']:
                st.success(f"Mặt cắt đập thỏa mãn điều kiện ổn định (K = {result['K']:.4f} ≥ Kc = {result['Kc']:.2f})")
            else:
                st.error(f"Mặt cắt đập KHÔNG thỏa mãn điều kiện ổn định (K = {result['K']:.4f} < Kc = {result['Kc']:.2f})")
            
            if result['sigma'] <= 0:
                st.success(f"Mặt cắt đập thỏa mãn điều kiện không kéo (σ = {result['sigma']:.4f} T/m² ≤ 0)")
            else:
                st.warning(f"Mặt cắt đập có ứng suất kéo ở mép thượng lưu (σ = {result['sigma']:.4f} T/m² > 0)")
            
            # Hiển thị thời gian tính toán
            st.info(f"Thời gian tính toán: {result['computation_time']:.2f} giây")
            
            # Tạo tabs cho các biểu đồ
            result_tabs = st.tabs(["Mặt cắt đập", "Biểu đồ hàm mất mát", "Xuất báo cáo"])
            
            # Tab mặt cắt đập
            with result_tabs[0]:
                # Tạo biểu đồ Plotly tương tác
                fig = create_force_diagram(result)
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab biểu đồ hàm mất mát
            with result_tabs[1]:
                # Tạo biểu đồ Plotly tương tác
                loss_fig = plot_loss_curve(result['loss_history'])
                st.plotly_chart(loss_fig, use_container_width=True)
            
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
                        'Số vòng lặp',
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
                        f"{selected_result['computation_time']:.2f} giây",
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
                    st.session_state.max_iterations = selected_result['iterations']
                    st.success("Đã tải thông số vào form tính toán. Chuyển sang tab 'Tính toán' để tiếp tục.")
    else:
        st.info("Chưa có kết quả tính toán nào được lưu trong cơ sở dữ liệu.")

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
    
    1. **Điều kiện ổn định trượt**: Hệ số ổn định K ≥ Kc
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
    
    1. Phạt nếu K < Kc (đảm bảo ổn định)
    2. Phạt nếu σ > 0 (đảm bảo không kéo)
    3. Tối thiểu hóa diện tích A
    
    #### Tài liệu tham khảo
    
    1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
    2. Nguyễn Văn Mạo, Đỗ Văn Bình (2010). Tính toán thiết kế đập bê tông trọng lực. NXB Xây dựng, Hà Nội.
    """)

# Tab Giới thiệu
with tabs[3]:
    st.markdown("""
    ### Giới thiệu
    
    Ứng dụng **Tính toán tối ưu mặt cắt đập bê tông trọng lực** là một công cụ chuyên nghiệp giúp kỹ sư và nhà thiết kế tìm ra mặt cắt kinh tế nhất cho đập bê tông trọng lực (phần không tràn) đồng thời đảm bảo các yêu cầu về ổn định và an toàn.
    
    #### Tính năng chính
    
    - **Tính toán tối ưu sử dụng PINNs**: Áp dụng mô hình mạng nơ-ron học sâu kết hợp với các ràng buộc vật lý để tìm mặt cắt đập tối ưu
    - **Giao diện người dùng**: Thiết kế tối giản, sạch sẽ theo phong cách Apple
    - **Trực quan hóa**: Hiển thị sơ đồ lực tác dụng và biểu đồ hàm mất mát tương tác
    - **Báo cáo**: Xuất báo cáo dạng Excel
    - **Cơ sở dữ liệu**: Lưu trữ và quản lý kết quả tính toán
    
    #### Hướng dẫn sử dụng
    
    1. Nhập các thông số đầu vào trong tab "Tính toán"
    2. Nhấn nút "Tính toán tối ưu" để thực hiện tính toán
    3. Xem kết quả tính toán và các biểu đồ trực quan
    4. Xuất báo cáo dạng Excel nếu cần
    5. Xem lịch sử tính toán trong tab "Lịch sử tính toán"
    
    #### Về tác giả
    
    Ứng dụng này được phát triển bởi nhóm nghiên cứu về ứng dụng trí tuệ nhân tạo trong kỹ thuật xây dựng công trình thủy lợi.
    
    #### Liên hệ
    
    Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ qua email: example@example.com
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Công cụ tính toán tối ưu mặt cắt đập bê tông trọng lực | Phiên bản PINNs 1.0</p>
</div>
""", unsafe_allow_html=True)

# Biểu đồ hàm mất mát
def plot_loss_curve(loss_history):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))),
        y=loss_history,
        mode='lines',
        line=dict(color='red', width=2),
        name='Hàm mất mát'
    ))

    fig.update_layout(
        title='Quá trình tối ưu hóa',
        xaxis_title='Số vòng lặp',
        yaxis_title='Giá trị hàm mất mát',
        width=800,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white'
    )

    return fig
