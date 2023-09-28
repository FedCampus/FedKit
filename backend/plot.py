import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from telemetry.models import EvaluateInsTelemetryData

import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation


x = []
y = []

xmax = 0
xmin = 0
ymax = 0
ymin = 0


def append(data_x, data_y):
    num = random.randint(1, 10)
    data_x.append(num)
    data_y.append(random.randint(1, 10))


fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Value")

(line,) = ax.plot(x, y)


# FuncAnimation动画回调函数
def update(i):
    y.append(EvaluateInsTelemetryData.objects.all().values_list("loss", flat=True)[i])
    x.append(i)

    # 更新图表数据
    line.set_xdata(x)
    line.set_ydata(y)
    ax.set_xlim(0, 100)
    ax.set_ylim(min(y) - 20, max(y))

    # 重绘图表
    fig.canvas.draw()


ani = animation.FuncAnimation(fig, update, interval=60)
plt.show()
# a = EvaluateInsTelemetryData.objects.all().values_list("loss", flat=True)
# print(len(a))
