import torch
import torch.nn as nn
import sys, os
sys.path.append(os.pardir)
sys.path.remove('/home/xiaohong/anaconda3/envs/pidnet/lib/python3.8/site-packages/cv2')
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from utils import CustomCosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt

initial_lr = 1e-5

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()

# optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
# scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1/(epoch+1))


optimizer = getattr(torch.optim, "AdamW")
optimizer = optimizer(net_1.parameters(), lr=initial_lr)
scheduler = CustomCosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=1000, cycle_mult=3, max_lr=1e-2, min_lr=1e-6,
                                                warmup_steps=500, gamma=0.707)

# scheduler = {
#             'scheduler': scheduler1,
#             'interval': "step",
#             'name': "learning rate"
#             }

print("初始化的学习率：", optimizer.defaults['lr'])

lr_list = []
t_lr = initial_lr
lr_list.append(t_lr)
step_list = [0]
total_batch = 15813
for epoch in range(0, 90):
    # train
    for batch in range(total_batch):
        optimizer.step()
        tmp_lr = optimizer.param_groups[0]['lr']
        # print("第%d个epoch, 第%d个batch,学习率：%f" % (epoch, batch, tmp_lr))
        if t_lr != tmp_lr:
            t_lr = tmp_lr
            lr_list.append(t_lr)
            step_list.append(epoch * total_batch + batch)
        scheduler.step()
    # print("----")

plt.plot(step_list, lr_list)
plt.xlabel("step")
plt.ylabel("lr")
plt.title("learning rate's curve changes as epoch goes on!")
plt.show()