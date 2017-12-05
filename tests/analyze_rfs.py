import numpy as np
import pandas as pd
import statsmodels.formula.api as sm


def process_data(f):
    """Process a file into a df."""
    data = np.load(f)
    on = ['on_' + m for m in data['rfs'][0]['on'].keys()]
    off = ['off_' + m for m in data['rfs'][0]['off'].keys()]
    all_c = on + off + ['perf', 'cell']
    rf_data = []
    for it_rf, dp in zip(data['rfs'], data['perf']):
        candidate_rf = [it_rf['on'].values() + it_rf['off'].values()]
        if None in candidate_rf[0]:
            t = 0  # simple
        else:
            t = 1  # complex
        candidate_rf = [0 if x is  None else x for x in candidate_rf[0]]
        rf_data += [candidate_rf + [dp] + [t]]
    return pd.DataFrame(rf_data, columns=all_c), all_c


files = [
    ### L2/3
    'ALLEN_files/filters/760_cells_2017_11_05_16_36_55_conv2d_recalc_False.npz',
    # 'ALLEN_files/filters/760_cells_2017_11_05_16_36_55_DoG_recalc_False.npz',
    # 'ALLEN_files/filters/760_cells_2017_11_05_16_36_55_sep_conv2d_recalc_False.npz',

    # 'ALLEN_files/filters/760_cells_2017_11_05_16_36_55_conv2d_recalc_True.npz',
    # 'ALLEN_files/filters/760_cells_2017_11_05_16_36_55_DoG_recalc_True.npz'
    # 'ALLEN_files/filters/760_cells_2017_11_05_16_36_55_sep_conv2d_recalc_True.npz',

    ### L4
    'ALLEN_files/filters/550_cells_2017_11_17_00_00_00_conv2d_recalc_False.npz',
    # 'ALLEN_files/filters/550_cells_2017_11_17_00_00_00_DoG_recalc_False.npz',
    # 'ALLEN_files/filters/550_cells_2017_11_17_00_00_00_sep_conv2d_recalc_False.npz',

    # 'ALLEN_files/filters/550_cells_2017_11_17_00_00_00_conv2d_recalc_True.npz',
    # 'ALLEN_files/filters/550_cells_2017_11_17_00_00_00_DoG_recalc_True.npz'
    # 'ALLEN_files/filters/550_cells_2017_11_17_00_00_00_sep_conv2d_recalc_True.npz',

]

reg_type = 'lr'
dfs, conds = [], []
stacks = []
for idx, f in enumerate(files):
    itd, all_c = process_data(f)
    df = (itd - itd.mean()) / itd.std()
    sit = np.hstack((df.as_matrix(), np.repeat(idx, df.shape[0]).reshape(-1, 1)))
    stacks += [sit]
    IV = [x for x in all_c if x != 'perf']
    IV = [x for x in IV if x != 'on_center_y']
    IV = [x for x in IV if x != 'on_center_x']
    IV = [x for x in IV if x != 'off_center_y']
    IV = [x for x in IV if x != 'off_center_x']
    IV = ' + '.join(IV)
    if reg_type == 'lr':
        result = sm.ols(formula='perf ~ %s' % IV, data=df).fit() 
    elif reg_type == 'mixed':
        result = smf.mixedlm(formula='Weight ~ Time', data=df, groups=df['model'])
    print 'R2 %s' % result.rsquared 
    # print result.summary()
    print 'F %s' % result.fvalue
    dfs += [result.tvalues.as_matrix()]
    conds += [result.tvalues.keys()]
res = {k: v for k, v in zip(conds[0], np.around(dfs[0] - dfs[1], 3))}
print res




