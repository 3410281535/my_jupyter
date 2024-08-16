import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_efficient_frontier(ax, sigma_P, r_f, E_r_P, sigma_C_range=None):
    """
    在给定的坐标轴上绘制期望收益与风险资产标准差之间的关系图。
    
    参数:
    ax -- 要绘制的坐标轴
    sigma_P -- 风险资产的标准差
    r_f -- 无风险收益率
    E_r_P -- 风险资产的期望收益率
    sigma_C_range -- 风险资产标准差范围的数组
    """
    if sigma_C_range is None:
        sigma_C_range = np.linspace(0, 0.3, 500)
    
    # 计算期望收益率
    E_r = r_f + (sigma_C_range / sigma_P) * (E_r_P - r_f)
    
    ax.clear()
    ax.plot(sigma_C_range, E_r, label='资本配置线', color='blue')

    # 绘制点F
    F = (0, r_f)
    ax.plot(*F, 'ro')  # 点F
    ax.annotate('F', xy=F, xytext=(F[0] + 0.01, F[1] + 0.01),
                arrowprops=dict(facecolor='red', shrink=0.05))
    ax.axhline(y=r_f, color='gray', linestyle='--', xmin=0.02, xmax=0.98)
    ax.text(0.02, r_f, 'r_f=7%', verticalalignment='bottom')

    # 绘制点P
    P = (sigma_P, E_r_P)
    ax.plot(*P, 'go')  # 点P
    ax.annotate('P', xy=P, xytext=(P[0] + 0.01, P[1] - 0.01),
                arrowprops=dict(facecolor='green', shrink=0.05))
    ax.axvline(x=sigma_P, color='gray', linestyle='--')
    ax.axhline(y=E_r_P, color='gray', linestyle='--')
    ax.text(sigma_P + 0.01, 0, f'$σ={sigma_P*100:.0f}%$\n$σ_P={sigma_P*100:.0f}%$\n$y=1$', verticalalignment='bottom')
    ax.text(sigma_P + 0.01, E_r_P, f'$E(r_P)={E_r_P*100:.0f}%$', verticalalignment='bottom')
    
    # 设置刻度
    ax.set_xticks(np.arange(0, 0.31, 0.01))
    ax.set_yticks(np.arange(r_f, E_r_P + 0.01, 0.01))
    
    ax.set_xlabel('风险资产标准差 $σ_C$')
    ax.set_ylabel('期望收益率 $E(r_c)$')
    ax.set_title('期望收益率与风险资产标准差的关系')
    ax.legend()
    ax.grid(True)

def interactive_plot_efficient_frontier():
    """
    创建交互式图形并在 Jupyter Notebook 中显示。
    """
    sigma_P_init = 0.22
    r_f = 0.07
    E_r_P = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    plot_efficient_frontier(ax, sigma_P=sigma_P_init, r_f=r_f, E_r_P=E_r_P)
    
    # 添加控件
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'σ_P', 0.01, 0.30, valinit=sigma_P_init, valstep=0.01, valfmt='%1.2f')
    
    def update(val):
        sigma_P = slider.val
        plot_efficient_frontier(ax, sigma_P=sigma_P, r_f=r_f, E_r_P=E_r_P)
        plt.draw()
    
    slider.on_changed(update)
    plt.show()