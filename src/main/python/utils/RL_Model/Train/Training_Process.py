import torch
import torch.nn as nn
import numpy as np
from src.main.python.common_func.logger import Log

def Update(model, stat, acts, log_prob, adv, rwd, OPT, ppo_steps, ppo_clip):
    PLOSS, VLOSS = 0, 0

    # 全部變數取消梯度
    stat = stat.detach()
    acts = acts.detach()
    log_prob = log_prob.detach()
    rwd = rwd.detach()
    for _ in range(ppo_steps):
        pred_prob, pred_rwd = model(stat) # 模型預測動作機率和回報值
        pred_rwd = pred_rwd.reshape(-1, ) # 維度從(N, 1)修改為(N , )

        ## 要用預測的機率去抽樣真正的動作
        dist = torch.distributions.Categorical(pred_prob) 
        new_log_prob = dist.log_prob(acts)

        ## PPO Loss的計算
        IS = (new_log_prob - log_prob).exp() # Importance sampling
        ploss = -torch.min(IS*adv, torch.clamp(IS, min = 1.0-ppo_clip, max = 1.0+ppo_clip)*adv).mean()
        vloss = nn.functional.smooth_l1_loss(rwd, pred_rwd).mean()
        ## 模型更新
        OPT.zero_grad()
        ploss.backward()
        vloss.backward()
        OPT.step()

        PLOSS += ploss.item() # 取值，該值為float
        VLOSS += vloss.item() # 取值，該值為float
    # print(PLOSS/ppo_steps, VLOSS/ppo_steps)

def fit(model, env, epochs):
    logger = Log(__name__)
    logger.write('Create Training Object')
    OPT = torch.optim.Adam(model.parameters(), lr = 0.0005)
    RWDS, PROBS = list(), list()
    state = torch.tensor(env.reset()[0]).to(torch.float32)
    logger.write('Start Training')
    for epoch in range(epochs):
        logger.write(f'Training in epoch {epoch + 1}')
        states, acts, log_probs, rwds, vals = list(), list(), list(), list(), list()
        step, rwd, done = 0, 0, False
        while not done:
            # 輸入環境狀態
            state = torch.tensor(state).to(torch.float32); states.append(state)
            # 計算policy機率和預測rewards
            pred_act, pred_rwd = model(state); pred_act = pred_act.view(-1)
            # 抽樣policy
            dist = torch.distributions.Categorical(pred_act)
            act = dist.sample()
            # 計算參數(for update)
            log_prob = dist.log_prob(act)
            state, temp_rwd, done, _, _ = env.step(int(act))
            log_probs.append(log_prob)
            rwds.append(temp_rwd); rwd += float(temp_rwd)
            acts.append(act); vals.append(torch.tensor(pred_rwd).to(torch.float32))
        states, acts, log_probs, vals = torch.stack(states), torch.stack(acts), torch.stack(log_probs), torch.stack(vals)
        # 計算報酬
        rets, ret = list(), 0
        for r in reversed(rwds):
            ret = r + ret*0.99
            rets.insert(0, ret)
        rets = torch.tensor(rets)
        rets = (rets - rets.mean())/rets.std()
        # 計算advantages
        advs = rets - vals
        advs = (advs - advs.mean())/advs.std()
        logger.write(f'PPO Update')
        Update(model, states, acts, log_probs, advs, rets, OPT, 5, 0.2)
        RWDS.append(rwd)
        print(f'EPOCH {epoch+1}, Rewards: {rwd}, Avg Rewards: {np.mean(RWDS)}')
        logger.write(f'Epoch {epoch + 1} Finished !')
    logger.write(f'Training Finished !')
    logger.write(f'Plot Learning Curve')
    import matplotlib.pyplot as plt
    plt.figure(figsize = (24, 3))
    plt.plot([i + 1 for i in range(epochs)], RWDS, label =  'Total Rewards')
    plt.grid()
    plt.legend()
    plt.show()
    logger.write(f'Plotting Finished !')

