import environ, data, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from analytic_tools import read_data, calculate_spread, test_fixed_stop

import time
import os
import enum
import torch


class Actions(enum.Enum):
    Short = 0
    Neutral = 1
    Long = 2


def run_model(
    spread_data,
    args_model,
    mean_spread,
    std_spread,
    stop_loss=4.0,
    N=20,
    commission=0.1,
    stop_loss_pct=5.0,
):
    env = environ.SpreadEnv(
        spread_data, bars_count=N, reset_on_close=False, random_ofs_on_reset=False
    )

    # net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)
    net = models.Actor(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(
        torch.load(args_model, map_location=lambda storage, loc: storage)
    )

    obs = env.reset()
    start_price = env._state._cur_close()

    # total_reward = 0.0
    step_idx = 0
    rewards = []
    actions_name = []
    positions = []
    open_prices = []
    close_prices = []
    open_times = []
    close_times = []
    position = 0
    open_price = 0

    while True:
        cur_price = env._state._cur_close()
        # if (position == -1 and cur_price > stop_loss) or (
        #     position == 1 and cur_price < -stop_loss
        # ):
        if cur_price > stop_loss or cur_price < -stop_loss:
            done = True
        else:
            step_idx += 1
            obs_v = torch.tensor([obs])
            action_probs = net(obs_v)
            dist = torch.distributions.Categorical(action_probs)
            # action_idx = dist.sample().numpy()[0] # random
            action_idx = dist.probs.argmax(dim=1).numpy()[0]  # deterministic
            action = Actions(action_idx)
            actions_name.append(action.name)
            obs, reward, done, _ = env.step(action_idx)
            
            if position == 0 and action == Actions.Long:
                open_price = env._state._cur_close()
                open_time = env._state._offset
                position = 1
                positions.append(position)
                open_prices.append(open_price)
                open_times.append(open_time)
            elif position == 0 and action == Actions.Short:
                open_price = env._state._cur_close()
                open_time = env._state._offset
                position = -1
                positions.append(position)
                open_prices.append(open_price)
                open_times.append(open_time)
            elif position == 1 and action != Actions.Long:
                close_price = env._state._cur_close()
                close_prices.append(close_price)
                position = 0
                close_time = env._state._offset
                close_times.append(close_time)
            elif position == -1 and action != Actions.Short:
                close_price = env._state._cur_close()
                close_prices.append(close_price)
                position = 0
                close_time = env._state._offset
                close_times.append(close_time)
            elif position !=0 : # keep the position
                paper_return = (cur_price - open_price) * position * std_spread * 100
                if paper_return < - stop_loss_pct:
                    done = True

        # rewards.append(reward)

        if done:
            if position != 0:
                close_price = env._state._cur_close()
                close_prices.append(close_price)
                position = 0
                close_time = env._state._offset
                close_times.append(close_time)
            break
    env.close()
    d = pd.DataFrame({"Position": positions})
    # convert the normalized spread to the  original spread
    d["OpenSpread"] = np.array(open_prices) * std_spread + mean_spread
    d["CloseSpread"] = np.array(close_prices) * std_spread + mean_spread
    d["Open_Z"] = np.array(open_prices)
    d["Close_Z"] = np.array(close_prices)
    d["Return(%)"] = (d.CloseSpread - d.OpenSpread) * d.Position * 100 - 2 * commission
    d["CumulativeReturn(%)"] = (np.cumprod((1 + d["Return(%)"] / 100)) - 1) * 100
    d["OpenTime"] = np.array(open_times, dtype=int)
    d["CloseTime"] = np.array(close_times, dtype=int)
    d["HoldingPeriod"] = d.CloseTime - d.OpenTime
    return np.round(d, 5)


# def compute_mean_std_spread(year):
#     pairs = pd.read_csv('pairs/{}-01-01_{}-12-31.csv'.format(year,year),index_col=0)
#     pairs = pairs[['Stock1','Stock2']].values#[:num_pairs]
#     trade_start = '{}-01-01'.format(year+1); trade_end = '{}-12-31'.format(year+1)
#     n_pairs = len(pairs)
#     spread_stat = np.zeros((n_pairs,3))

#     for i,pair in enumerate(pairs):
#         ric1, ric2 = pair
#         df,_ = read_data([ric1,ric2],'30min')
#         formation_start = '{}-01-01'.format(year); formation_end = '{}-12-31'.format(year)
#         trade_start = '{}-01-01'.format(year+1); trade_end = '{}-12-31'.format(year+1)
#         df_f = df[formation_start:formation_end]
#         df_t = df[trade_start:trade_end]
#         _,_,mean_spread,std_spread,h = calculate_spread(df_f,df_t)
#         spread_stat[i] = mean_spread, std_spread, h
#     spread_stat = pd.DataFrame(spread_stat,columns=['Mean','Std','HedgeRatio'],index=np.arange(1,n_pairs+1))
#     df = pd.DataFrame(pairs,columns=['Stock1','Stock2'],index=np.arange(1,n_pairs+1))
#     df = pd.concat([df,spread_stat],axis=1)
#     df.to_csv('pairs_spread_info/{}.csv'.format(year))

# compute_mean_std_spread(2009)
# compute_mean_std_spread(2010)
# compute_mean_std_spread(2011)
# compute_mean_std_spread(2012)


def get_mean_std_spread(i, year):
    df = pd.read_csv("pairs_spread_info/{}.csv".format(year), index_col=0)
    mean_spread, std_spread, _ = df.iloc[i][["Mean", "Std", "HedgeRatio"]]
    return mean_spread, std_spread


def test_year(year, model_param, check_points, U=1.5, fee=0.05, SL=50):
    model = "saves/{}-".format(year)
    model += model_param
    results_dir = "results/{}_{}_SL{}/".format(year, model_param, SL)

    # in-sample
    pairs = pd.read_csv("pairs/{}-01-01_{}-12-31.csv".format(year, year), index_col=0)
    pairs = pairs[["Stock1", "Stock2"]].values  # [:num_pairs]
    trade_start = "{}-01-01".format(year + 1)
    trade_end = "{}-12-31".format(year + 1)
    print(
        "In-sample {} to {}  cp{}-{}".format(
            trade_start, trade_end, check_points[0], check_points[-1]
        )
    )
    for i in range(len(pairs)):
        start_time = time.time()
        mean_spread, std_spread = get_mean_std_spread(i, year)
        ric1, ric2 = pairs[i]
        spread_data = "data/spread_t_{}_{}_30min_{}_{}.csv".format(
            ric1, ric2, trade_start, trade_end
        )
        spread_data = {"{}_{}".format(ric1, ric2): data.load_spreads(spread_data)}
        csv_dir = results_dir + "{}_{}/{}_{}".format(trade_start, trade_end, ric1, ric2)
        os.makedirs(csv_dir, exist_ok=True)
        for cp in check_points:
            model_cp = model + "/checkpoint-{:3d}.data".format(cp)
            results = run_model(
                spread_data,
                model_cp,
                mean_spread,
                std_spread,
                stop_loss=SL,
                commission=fee,
            )
            results.to_csv(csv_dir + "/CP{}.csv".format(cp))
        elapsed_time = time.time() - start_time
        print(
            "{} Testing {} ... Time: {}".format(
                i + 1, pairs[i], time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            )
        )

    # out-of-sample
    pairs = pd.read_csv(
        "pairs/{}-01-01_{}-12-31.csv".format(year + 1, year + 1), index_col=0
    )
    pairs = pairs[["Stock1", "Stock2"]].values  # [:num_pairs]
    trade_start = "{}-01-01".format(year + 2)
    trade_end = "{}-12-31".format(year + 2)
    print("Out-of-sample {} to {}".format(trade_start, trade_end))
    for i in range(len(pairs)):
        print("{} Testing {} ...".format(i + 1, pairs[i]))
        mean_spread, std_spread = get_mean_std_spread(
            i, year + 1
        )  # previously used year instead of year+1
        ric1, ric2 = pairs[i]
        spread_data = "data/spread_t_{}_{}_30min_{}_{}.csv".format(
            ric1, ric2, trade_start, trade_end
        )
        spread_data = {"{}_{}".format(ric1, ric2): data.load_spreads(spread_data)}
        csv_dir = results_dir + "{}_{}/{}_{}".format(trade_start, trade_end, ric1, ric2)
        os.makedirs(csv_dir, exist_ok=True)
        for cp in check_points:
            model_cp = model + "/checkpoint-{:3d}.data".format(cp)
            results = run_model(
                spread_data,
                model_cp,
                mean_spread,
                std_spread,
                stop_loss=SL,
                commission=fee,
            )
            results.to_csv(csv_dir + "/CP{}.csv".format(cp))
    print("----- done -----")


def read_results_in(year, model_param, check_points, U=1.5, fee=0.05, SL=50):
    all_returns = []
    all_trades = []
    results_dir = "results/{}_{}_SL{}/".format(year, model_param, SL)
    pairs = pd.read_csv("pairs/{}-01-01_{}-12-31.csv".format(year, year), index_col=0)
    pairs = pairs[["Stock1", "Stock2"]].values  # [:num_pairs]
    trade_start = "{}-01-01".format(year + 1)
    trade_end = "{}-12-31".format(year + 1)
    for i in range(len(pairs)):
        ric1, ric2 = pairs[i]
        results_path = results_dir + "{}_{}/{}_{}".format(
            trade_start, trade_end, ric1, ric2
        )
        returns = []
        trades = []
        for cp in check_points:
            df = pd.read_csv(results_path + "/CP{}.csv".format(cp), index_col=0)
            if len(df) == 0:
                returns.append(0)
            else:
                returns.append(df["CumulativeReturn(%)"].iloc[-1])
            trades.append(len(df))

        returns = np.array(returns)

        all_returns.append(returns)
        all_trades.append(trades)

    all_returns = np.array(all_returns)
    all_trades = np.array(all_trades)
    return all_returns, all_trades


def read_results_out(year, model_param, check_points, U=1.5, fee=0.05, SL=50):
    all_returns = []
    all_trades = []
    results_dir = "results/{}_{}_SL{}/".format(year, model_param, SL)
    pairs = pd.read_csv(
        "pairs/{}-01-01_{}-12-31.csv".format(year + 1, year + 1), index_col=0
    )
    pairs = pairs[["Stock1", "Stock2"]].values  # [:num_pairs]
    trade_start = "{}-01-01".format(year + 2)
    trade_end = "{}-12-31".format(year + 2)
    for i in range(len(pairs)):
        ric1, ric2 = pairs[i]
        results_path = results_dir + "{}_{}/{}_{}".format(
            trade_start, trade_end, ric1, ric2
        )
        returns = []
        trades = []
        for cp in check_points:
            df = pd.read_csv(results_path + "/CP{}.csv".format(cp), index_col=0)
            if len(df) == 0:
                returns.append(0)
            else:
                returns.append(df["CumulativeReturn(%)"].iloc[-1])
            trades.append(len(df))

        returns = np.array(returns)

        all_returns.append(returns)
        all_trades.append(trades)

    all_returns = np.array(all_returns)
    all_trades = np.array(all_trades)
    return all_returns, all_trades


def plot_year_beta(year, model_param, check_points, SL=50):
    # U:1.5 fee:0.05 SL:50
    # fig_temp_dir = "results_r4_2y_all/U1.5_Fee0.05_SL{}/figs_temp/".format(SL)
    fig_temp_dir = "results/figs/"
    os.makedirs(fig_temp_dir, exist_ok=True)
    in_returns, in_trades = read_results_in(year, model_param, check_points, SL=SL)
    out_returns, out_trades = read_results_out(year, model_param, check_points, SL=SL)

    plt.figure(figsize=(14, 8))
    # sl = "None" if SL > 20 else SL
    beta = model_param.split("-")[-1][1:]
    subtitle_str = "Performance of RL-{} during {} (training) - {} (testing)".format(
        beta, year + 1, year + 2
    )
    if SL <= 20:
        subtitle_str += f"\nStop-loss: {SL}"
    plt.suptitle(subtitle_str, fontsize=14)
    plt.subplot(221)
    plt.plot(check_points / 10, in_returns.mean(axis=0))
    #plt.axhline(0, ls="--", c="grey")
    plt.grid(True)
    plt.title("Average Cumulative Return (Training)")
    plt.ylabel("Return (%)")

    plt.subplot(222)
    plt.plot(check_points / 10, out_returns.mean(axis=0))
    #plt.axhline(0, ls="--", c="grey")
    plt.grid(True)
    plt.title("Average Cumulative Return (Testing)")
    plt.ylabel("Return (%)")

    plt.subplot(223)
    plt.plot(check_points / 10, in_trades.mean(axis=0))
    plt.title("Average Trades (Training)")
    plt.ylabel("Trades")
    plt.xlabel("Training Steps " + r"$(\times 10^6$)")
    plt.grid(True)

    plt.subplot(224)
    plt.plot(check_points / 10, out_trades.mean(axis=0))
    plt.title("Average Trades (Testing)")
    plt.ylabel("Trades")
    plt.xlabel("Training Steps " + r"$(\times 10^6$)")
    plt.grid(True)
    plt.savefig(fig_temp_dir + "{}_{}_SL{}.jpg".format(year, model_param, SL), dpi=300)
    # plt.show()


if __name__ == "__main__":
    check_points = np.arange(0, 1001, 20)
    model_param = "2010-C0.005-H0.001-L5e-06-T2048-B64-N5-E0.001-b0.0"
    year = int(model_param[:4])
    model_param = model_param[5:]
    stop_loss = 50
    test_year(year,model_param,check_points, SL=stop_loss)
    plot_year_beta(year, model_param, check_points, SL=stop_loss)
