import pandas as pd
import matplotlib.pyplot as plt


TYPE = 'open_ret'
BM_NUM = 1800

SIZE_RANK = 'D:/FinData/EQ/size_rank.csv'
OPEN_RET = 'D:/FinData/EQ/'+TYPE+'.csv'
OUTPUT_PATH = 'output_'+TYPE+'/'


if __name__ == '__main__':
    print('Loading Size Rank ...')
    size_rank = pd.read_csv(SIZE_RANK, index_col=[0,1], date_format="%Y.%m.%d").squeeze()

    print('Loading Open Return ...')
    open_ret = pd.read_csv(OPEN_RET, index_col=[0,1], date_format="%Y.%m.%d").squeeze()

    print('Generating Benchmark ...')
    bm = size_rank.where(size_rank <= BM_NUM).dropna().to_frame()
    bm = bm.join(open_ret, how='inner').dropna()

    bm_ret = bm.groupby(['date'])['ret'].sum() / BM_NUM
    bm_ret.to_csv(OUTPUT_PATH + 'ret_bm_'+str(BM_NUM)+'.csv')

    bm_cumret = (1 + bm_ret).cumprod() - 1
    bm_cumret.plot(grid=True, title='Benchmark Cumulative Return')
    plt.show()