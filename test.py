import torch
# from collections import namedtuple
#
# Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
# T1 = Transition('A1', 'B1', 'C1', 'D1')
# T2 = Transition('A2', 'B2', 'C2', 'D2')
# T3 = Transition('A3', 'B3', 'C3', 'D3')
# transitions = (T1, T2, T3)
# print(transitions)
# batch = Transition(*zip(*transitions))
# print(batch)
# #
# # state_batch = torch.stack(batch.state)
# # print(state_batch)
#
# # # 假设是时间步T1的输出
# # T1 = torch.tensor([[1, 2, 3],
# #                 [4, 5, 6],
# #                 [7, 8, 9]])
# # # 假设是时间步T2的输出
# # T2 = torch.tensor([[10, 20, 30],
# #                 [40, 50, 60],
# #                 [70, 80, 90]])
# # print(torch.stack((T1,T2),dim=0))
# # print(torch.stack((T1,T2),dim=1))
# # print(torch.stack((T1,T2),dim=2))

q = torch.tensor([[1, 2]])
print(q)
print(q.max(1))