import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Output, VBox, HBox
from IPython.display import display
from matplotlib import rcParams


"""
有时拉条拉完会有在两组值间反复跳动, 可用点击拉条来解决
"""

rcParams['font.family'] = 'SimHei'  
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题

def U_func(E_r, sigma, A):
    return E_r - 0.5 * A * sigma**2


def E_r(sigma, A, U_fixed):
    return U_fixed + 0.5 * A * sigma**2


# 绘制无差异曲线及当前点
def indifference_curve(sigma_current, E_r_current, A_current):
    plt.figure(figsize=(8, 6))

    sigma_values = np.linspace(0, 2, 400)

    # 计算当前A对应的U_fixed值
    U_fixed = U_func(1, 0.5, A_current)

    # 计算当前A的无差异曲线
    E_r_values = E_r(sigma_values, A_current, U_fixed)
    plt.plot(sigma_values, E_r_values, label=f'无差异曲线 (A={A_current})')

    plt.scatter(sigma_current, E_r_current, color='red', zorder=5, label='当前点')

    plt.axhline(y=E_r_current, color='gray', linestyle='--')
    plt.axvline(x=sigma_current, color='gray', linestyle='--')

    plt.xlim(0, 2.5)
    plt.ylim(-0.5, 2.5)


    plt.xlabel('$\sigma$')
    plt.ylabel('$E(r)$')
    plt.title('无差异曲线 $U = E(r) - \\frac{1}{2} A \\sigma^2$')


    plt.legend()

    plt.grid(True)

    plt.text(sigma_current + 0.1, E_r_current, f'U = {U_fixed:f}', fontsize=12, color='blue', verticalalignment='bottom')
    

    plt.show()

# 更新函数
def update_plot(sigma_current, E_r_current, A_current):
    # 更新E_r保持在无差异曲线上
    E_r_current = E_r(sigma_current, A_current, U_func(1, 0.5, A_current))

    sigma_current = np.clip(sigma_current, 0, 2)
    E_r_current = np.clip(E_r_current, -0.5, 2.5)

    E_r_slider.value = E_r_current  # 更新滑块值

    out.clear_output(wait=True)  # 清空当前输出
    with out:
        indifference_curve(sigma_current, E_r_current, A_current)

# 控件
sigma_slider = FloatSlider(min=0, max=2, value=0.5, description='$\sigma$', orientation='horizontal')
E_r_slider = FloatSlider(min=-0.5, max=2.5, value=1, description='$E(r)$', orientation='horizontal')
A_slider = FloatSlider(min=-3, max=10, value=5, step=0.1, description='$A$', orientation='horizontal')

# 计算初始值
def update_initial_A(A):
    global U_fixed
    U_fixed = U_func(1, 0.5, A)

# 输出区域
out = Output()

# 显示控件和输出
def init_widgets():
    display(VBox([sigma_slider, E_r_slider, A_slider, out]))

    # 绑定控件变化
    def on_sigma_change(change):
        update_plot(sigma_slider.value, E_r_slider.value, A_slider.value)

    def on_E_r_change(change):
        # 更新相应sigma
        sigma_current = np.sqrt(2 * (E_r_slider.value - U_func(1, 0.5, A_slider.value)) / A_slider.value)
        sigma_slider.value = np.clip(sigma_current, 0, 2)  # 确保在范围内
        update_plot(sigma_slider.value, E_r_slider.value, A_slider.value)

    def on_A_change(change):
        A_current = np.round(A_slider.value, 1)
        update_initial_A(A_current)
        update_plot(sigma_slider.value, E_r_slider.value, A_current)

    sigma_slider.observe(on_sigma_change, names='value')
    E_r_slider.observe(on_E_r_change, names='value')
    A_slider.observe(on_A_change, names='value')

    
    update_plot(sigma_slider.value, E_r_slider.value, A_slider.value)

# 初始化控件
init_widgets()