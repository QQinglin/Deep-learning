import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义爱心曲线函数
def f(x, t):
    return (np.abs(x) ** (2/3)) + 0.8 * np.sqrt(np.maximum(3.3 - x**2, 0)) * np.sin(t * np.pi * x)

# 设置 x 取值范围
x = np.linspace(-2, 2, 1000)  # 增加点数让曲线更平滑

# 创建图像
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 2.5)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title("Zhengdong's Heart Animation")
ax.grid(True)

# 初始化空的线条
line, = ax.plot([], [], 'r', linewidth=1, label=r'$f(x)$')
ax.legend()

# 更新函数，每一帧更新 t 的值
def update(frame):
    t = frame / 10  # 让 t 从 0 逐渐增加到 15
    y = f(x, t)
    line.set_data(x, y)
    return line,

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# 保存或显示动画
plt.show()

