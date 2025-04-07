# Tối ưu hóa triển khai PINNs cho tính toán mặt cắt đập bê tông

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

# Hàm tính vật lý dùng PINNs
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

# Hàm mất mát cải tiến
def loss_function(sigma, K, A, Kc, alpha):
    # Điều kiện ổn định trượt: K=Kc (không phải K>=Kc)
    # Sử dụng MSE để K tiến gần đến Kc thay vì chỉ đảm bảo K>=Kc
    penalty_K = (K - Kc)**2
    
    # Điều kiện ứng suất mép thượng lưu: σ≈0 (không có ứng suất kéo)
    # Phạt nặng nếu sigma > 0, nhưng cũng khuyến khích sigma tiến gần đến 0
    penalty_sigma = torch.where(sigma > 0, 
                               100 * sigma**2,  # Phạt nặng nếu có ứng suất kéo
                               (sigma - 0)**2)  # Khuyến khích sigma tiến gần đến 0
    
    # Tối thiểu hóa diện tích mặt cắt A
    # Đây là hàm mục tiêu chính cần tối ưu
    objective = alpha * A
    
    # Tổng hợp các thành phần
    return penalty_K.mean() + penalty_sigma.mean() + objective.mean()

# Hàm tối ưu hóa cải tiến với cơ chế hội tụ sớm
def optimize_dam_section(H, gamma_bt, gamma_n, f, C, Kc, a1, max_iterations=5000, convergence_threshold=1e-6, patience=50):
    alpha = 0.01  # hệ số phạt diện tích
    model = OptimalParamsNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)

    data = torch.ones((1, 1), device=device)
    
    # Lưu lịch sử tối ưu hóa
    loss_history = []
    best_loss = float('inf')
    best_params = None
    patience_counter = 0
    
    for epoch in range(max_iterations):
        optimizer.zero_grad()
        n, m, xi = model(data)
        sigma, K, A = compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1)[:3]
        loss = loss_function(sigma, K, A, Kc, alpha)
        loss.backward()
        optimizer.step()
        
        # Cập nhật learning rate
        scheduler.step(loss)
        
        # Lưu lịch sử loss
        current_loss = loss.item()
        loss_history.append(current_loss)
        
        # Kiểm tra điều kiện hội tụ
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = (n.detach().clone(), m.detach().clone(), xi.detach().clone())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Kiểm tra điều kiện dừng sớm
        if patience_counter >= patience:
            print(f"Dừng sớm tại epoch {epoch} do không cải thiện sau {patience} vòng lặp")
            break
            
        # Kiểm tra hội tụ dựa trên giá trị loss
        if epoch > 100 and abs(loss_history[-1] - loss_history[-100]) < convergence_threshold:
            print(f"Đã hội tụ tại epoch {epoch} với sai số {abs(loss_history[-1] - loss_history[-100])}")
            break
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Sử dụng tham số tốt nhất nếu có
    if best_params is not None:
        n, m, xi = best_params
    else:
        model.eval()
        n, m, xi = model(data)
        
    # Tính toán các đại lượng vật lý với tham số tối ưu
    sigma, K, A, G, W1, W2, Wt, Fct, Fgt, B, M0, G1, G2, W2_1, W2_2, lG1, lG2, lt, l2, l22, l1, P = compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1)
    
    # Tính độ lệch tâm
    e = B/2 - M0/P
    
    # Số vòng lặp thực tế đã thực hiện
    actual_iterations = epoch + 1

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
        'iterations': actual_iterations,  # Số vòng lặp thực tế
        'max_iterations': max_iterations, # Số vòng lặp tối đa
        'loss_history': loss_history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# Hàm tạo biểu đồ mặt cắt đập và sơ đồ lực
def create_force_diagram(result):
    """
    Tạo biểu đồ mặt cắt đập và sơ đồ lực
    
    Parameters:
    -----------
    result : dict
        Kết quả tính toán từ hàm optimize_dam_section
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Biểu đồ Plotly tương tác
    """
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

    # Tính vị trí đặt lực
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

# Hàm tạo biểu đồ hàm mất mát
def plot_loss_curve(loss_history):
    """
    Tạo biểu đồ hàm mất mát từ lịch sử tối ưu hóa
    
    Parameters:
    -----------
    loss_history : list
        Danh sách giá trị hàm mất mát theo từng vòng lặp
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Biểu đồ Plotly tương tác
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Tạo mảng chỉ số vòng lặp
    epochs = np.arange(len(loss_history))
    
    # Tạo biểu đồ
    fig = go.Figure()
    
    # Thêm đường biểu diễn hàm mất mát
    fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_history,
        mode='lines',
        name='Hàm mất mát',
        line=dict(color='#0066cc', width=2)
    ))
    
    # Cấu hình chung
    fig.update_layout(
        title='Biểu đồ hàm mất mát theo vòng lặp',
        xaxis_title='Vòng lặp',
        yaxis_title='Giá trị hàm mất mát',
        width=850,
        height=500,
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # Thêm lưới
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        type='log'  # Sử dụng thang logarit cho trục y
    )
    
    return fig
