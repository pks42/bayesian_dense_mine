import pandas as pd
import numpy as np
from scipy.stats import zscore

click = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_tfidf_counts_top_5_comps.tsv', sep = '\t', encoding = 'utf-8')
top_freq = click.iloc[:,1].tolist()

comps = list()
comp_num = [0]
for i in range(len(top_freq)):
    if top_freq[i] > 500:
        if click.iloc[i,2][0] == 'i':
            comps.append(click.iloc[i,2])
            comp_num.append(i+1)
            #print(click.iloc[i,2])


comps.remove('interactie.voorwaarden.akkoord')
comps.remove('interactie.voorwaarden.toestemming-vink-aan')
comp_num.remove(184)
comp_num.remove(185)
        
cust_intr = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_tfidf_one_hot_top_5_comps.tsv', sep = '\t', encoding = 'utf-8')
cust_sel_intr = cust_intr.iloc[:,comp_num]
head_out = list(cust_sel_intr)
cust_out = cust_sel_intr.set_index('cust_ID').T.to_dict('list')
cust_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\cust_data_cleaned_test_ready.tsv', sep = '\t', encoding = 'utf-8')
head_in = list(cust_data)
cust = cust_data.set_index('cust_ID').T.to_dict('list')

data = list()
data.append(head_in[1:])
data[0].extend(head_out[1:])
for i in cust_out:
    temp = cust[i]
    temp.extend(cust_out[i])
    data.append(temp)

nn_data = pd.DataFrame(data[1:], columns=data[0])
nn_data.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_combo_nn_data.tsv', sep ='\t', encoding='utf-8', index=False)

def create_sep_dict(values):
    sep_dict = dict()
    inc = 1.0/(len(values)-1)
    for i in range(len(values)):
        sep_dict[values[i]] = inc*(i)
    return sep_dict

def compute_intermediate(x, high, low):
    y = (x-low)/(high-low)
    return y

def dict_apply(x, norm_dict):
    return norm_dict[x]

def normalize_cust_data(data, cust_data):
    s_norm_dict = create_sep_dict([1,2])
    m_norm_dict = create_sep_dict([1,2,3,4,5])
    p_norm_dict = create_sep_dict([20,35,36])
    data['sex'] = data['sex'].apply(lambda x: dict_apply(x, s_norm_dict))
    data['marital'] = data['marital'].apply(lambda x: dict_apply(x, m_norm_dict))
    data['pension_status'] = data['pension_status'].apply(lambda x: dict_apply(x, p_norm_dict))
    age_max = (cust_data['age'].max()) + 1
    age_min = (cust_data['age'].min()) - 1
    data['age'] = data['age'].apply(lambda x: compute_intermediate(x, age_max, age_min))
    data['salary'] = data['salary'].round()
    data['part_time%'] = data['part_time%'].round()
    data['pension_rel_salary'] = data['pension_rel_salary'].round()
    salary_max = (cust_data['salary'].max() + 1)
    pension_rel_salary_max = (cust_data['pension_rel_salary'].max() + 1)
    data['salary'] = data['salary'].apply(lambda x: x/salary_max)
    data['part_time%'] = data['part_time%'].apply(lambda x: x/100.0)
    data['pension_rel_salary'] = data['pension_rel_salary'].apply(lambda x: x/pension_rel_salary_max)
    return data

def gauss_normalize_cust_data(data, cust_data):
    coln_names = ['age','sex','marital_1', 'marital_2', 'marital_3', 'marital_4', 'pension_status_20', 'pension_status_35', 'salary', 'part_time%', 'pension_rel_salary', 'newsletter']
    normed_data = pd.DataFrame(columns=coln_names)
    s_norm_dict = create_sep_dict([1,2])
    p_norm_dict = create_sep_dict([20,35,36])
    normed_data['sex'] = data['sex'].apply(lambda x: dict_apply(x, s_norm_dict))
    normed_data = normed_data.fillna(0)
    normed_data.loc[(data['marital']==1),'marital_1'] = 1.0
    normed_data.loc[(data['marital']==2),'marital_2'] = 1.0
    normed_data.loc[(data['marital']==3),'marital_3'] = 1.0
    normed_data.loc[(data['marital']==4),'marital_4'] = 1.0
    normed_data.loc[(data['pension_status']==20),'pension_status_20'] = 1.0
    normed_data.loc[(data['pension_status']==35),'pension_status_35'] = 1.0
    normed_data['age'] = zscore(data['age'])
    normed_data['age'] = (normed_data['age'] + abs(normed_data['age'].min()))/6.0 #Divison by 6.0 assuming mostly covered in 6 standard deviations
    normed_data['salary'] = zscore(data['salary'].round())
    normed_data['salary'] = (normed_data['salary'] + abs(normed_data['salary'].min()))/6.0
    normed_data['part_time%'] = zscore(data['part_time%'].round())
    normed_data['part_time%'] = (normed_data['part_time%'] + abs(normed_data['part_time%'].min()))/6.0
    normed_data['pension_rel_salary'] = zscore(data['pension_rel_salary'].round())
    normed_data['pension_rel_salary'] = (normed_data['pension_rel_salary'] + abs(normed_data['pension_rel_salary'].min()))/6.0
    n = len(data.iloc[0,:]) - (len(normed_data.iloc[0,:]) - 4)
    normed_data = pd.concat([normed_data, data.iloc[:,-n:]], axis = 1)
    return normed_data


nn_data1 = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_combo_nn_data.tsv', sep ='\t', encoding='utf-8')
del nn_data1['emp_reg_no']
# nn_data1 = normalize_cust_data(nn_data1, cust_data)
nn_data1 = gauss_normalize_cust_data(nn_data1, cust_data)

nn_data1.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_normalized_combo_nn_data.tsv', sep ='\t', encoding='utf-8', index = False)

# REMOVE EMPTY AND MULTIPLE OUTPUTS
nn_data2 = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_normalized_combo_nn_data.tsv', sep ='\t', encoding='utf-8')


head_out
nn_data2[head_out[1:]].sum()
nn_data2.iloc[~(nn_data2[head_out[1:]].sum == 1.0)]

for i in range(len(nn_data2.iloc[:,0])):
    if nn_data2.iloc[i,8:].sum() != 1.0:
        nn_data2.iloc[i,0] = 0.0
    
nn_data3 = nn_data2[nn_data2.age != 0.0]
nn_data3.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_single_op_normalized_combo_nn_data.tsv', sep ='\t', encoding='utf-8', index = False)

##CREATING FAKE DATASET
import numpy as np
import pandas as pd
a = 0.25*np.random.rand(872,1) + 0.5
a_o = 0.25*np.random.rand(872,1) + 0.0
a = np.append(a,a_o, axis = 0)
np.amin(a)
np.amax(a)
b = 0.25*np.random.rand(872,1) + 0.25
b_o = 0.25*np.random.rand(872,1) + 0.75
b = np.append(b,b_o, axis = 0)
np.amin(b)
np.amax(b)
a_y = np.ones((1744,1))
b_y = np.zeros((1744,1))
a = np.append(a,a_y, axis = 1)
b = np.append(b, b_y, axis = 1)
a_b = np.append(a,b, axis = 0)
np.random.shuffle(a_b)
fake_data = pd.DataFrame(a_b)
fake_data.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_single_op_FAKE_non_linear_data.tsv', sep ='\t', encoding='utf-8', index = False)

a = 0.5*np.random.rand(1744,1) + 0.5
np.amin(a)
np.amax(a)
b = 0.5*np.random.rand(1744,1)
np.amin(b)
np.amax(b)
a_y = np.ones((1744,1))
b_y = np.zeros((1744,1))
a = np.append(a,a_y, axis = 1)
b = np.append(b, b_y, axis = 1)
a_b = np.append(a,b, axis = 0)
np.random.shuffle(a_b)
fake_data = pd.DataFrame(a_b)
fake_data.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_single_op_FAKE_linear_data.tsv', sep ='\t', encoding='utf-8', index = False)

cust = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_combo_nn_data.tsv', sep = '\t', encoding = 'utf-8')
errs4 = np.ones((1,4))
for i in range(len(cust.iloc[:,0])):
    if cust.iloc[i,5] == 100 and cust.iloc[i,3] == 35 and cust.iloc[i,4] != 0:
        temp = np.array([i, cust.iloc[i,3], cust.iloc[i,4], cust.iloc[i,6]]).reshape((1,4))
        print(temp)
        errs4 = np.append(errs, temp, axis = 0)

print(errs3.shape)
print(temp)


##CREATING CLUSTERS
combo_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_normalized_combo_nn_data.tsv', sep = '\t', encoding = 'utf-8')
combo_data_drop = combo_data.drop(columns = ['salary'])

data_like = np.ones((1,13))
for i in range(len(combo_data_drop.iloc[:,0])):
    # tot = combo_data_drop.iloc[i,0]*2 + combo_data_drop.iloc[i,1]*5 + combo_data_drop.iloc[i,2]*3 + np.around(combo_data_drop.iloc[i,3],2) + np.around(combo_data_drop.iloc[i,4],5) + combo_data_drop.iloc[i,5]
    tot = combo_data_drop.iloc[i,0] + combo_data_drop.iloc[i,1] + (combo_data_drop.iloc[i,2] + combo_data_drop.iloc[i,3] + combo_data_drop.iloc[i,4] + combo_data_drop.iloc[i,5]) * 2.0 + (combo_data_drop.iloc[i,6] + combo_data_drop.iloc[i,7]) * 3.0 + np.around(combo_data_drop.iloc[i,8],2) + np.around(combo_data_drop.iloc[i,9],5) + combo_data_drop.iloc[i,10]
    ind = np.where(np.around(data_like[:,1],5) == np.around(tot,5))[0]
    if ind.size == 0:
        temp = np.zeros((1,13))
        temp[0,0] = i
        temp[0,1] = tot
        data_like = np.append(data_like, temp, axis = 0)
        ind = np.array([len(data_like)-1])
    data_like[ind[0],-1] += 1
    for j in range(2, 12, 1):
        data_like[ind[0],j] += combo_data_drop.iloc[i,(9+j)]

for i in range(len(data_like)):
    data_like[i,2:12] /= data_like[i,-1]



like_d = pd.DataFrame(data_like[1:,:])
like_d.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_aged_data_likelihood.tsv', sep = '\t', encoding = 'utf-8', index = False)


t = np.float64(((1,2,3),(5,6,7)))
i = np.where(t[:,0] == 10)
if i[0].size >0:
    print(i[0][0])

combo_data_drop.iloc[143,:]


combo_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_normalized_combo_nn_data.tsv', sep = '\t', encoding = 'utf-8')
data_like = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_aged_data_likelihood.tsv', sep = '\t', encoding = 'utf-8').values
def ret_data_like(x, data, i):
    ind = np.where(np.around(data[:,1],5) == np.around(x,5))
    return data[ind[0][0],i]

interests = comp_num[1:]
i = 2
for j in interests:
    # combo_data['data_like_'+str(j)] = combo_data['sex']*2 + combo_data['marital']*5 + combo_data['pension_status']*3 + np.around(combo_data['part_time%'],2) + np.around(combo_data['pension_rel_salary'],5) + combo_data['newsletter']
    combo_data['data_like_'+str(j)] = combo_data_drop.iloc[:,0] + combo_data_drop.iloc[:,1] + (combo_data_drop.iloc[:,2] + combo_data_drop.iloc[:,3] + combo_data_drop.iloc[:,4] + combo_data_drop.iloc[:,5]) * 2.0 + (combo_data_drop.iloc[:,6] + combo_data_drop.iloc[:,7]) * 3.0 + np.around(combo_data_drop.iloc[:,8],2) + np.around(combo_data_drop.iloc[:,9],5) + combo_data_drop.iloc[:,10]
    combo_data['data_like_'+str(j)] = combo_data['data_like_'+str(j)].apply(lambda x: ret_data_like(x, data_like, i))
    i += 1

combo_data.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_aged_normalized_combo_nn_data.tsv', sep = '\t', encoding = 'utf-8', index = False)

combo_data_drop = combo_data.drop(columns = ['y_54','y_55','y_69','y_81','y_98','y_99','y_107','y_131','y_161','data_like_54','data_like_55','data_like_69','data_like_81','data_like_98','data_like_99','data_like_107','data_like_131','data_like_161'])
t = combo_data_drop.loc[combo_data_drop['y_157']==1.0]
t.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\y_157_true.tsv', sep = '\t', encoding = 'utf-8')


data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\cust_data_cleaned_test_ready.tsv', sep='\t', encoding = 'utf-8')
data['marital'].value_counts()


##CORRECTED DATA LIKELIHOOD
combo_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_aged_normalized_combo_nn_data.tsv', sep='\t', encoding='utf-8')
rec_count = dict()
rec_pos = dict()
combo_data.iloc[:5,30]
for i in range(len(combo_data)):
    rec_id = str(combo_data.iloc[i,0].round(7)) + str(combo_data.iloc[i,1]) + str(combo_data.iloc[i,2]) + str(combo_data.iloc[i,3]) + str(combo_data.iloc[i,4]) + str(combo_data.iloc[i,5]) + str(combo_data.iloc[i,6]) + str(combo_data.iloc[i,7]) + str(combo_data.iloc[i,8].round(7)) + str(combo_data.iloc[i,9].round(7)) + str(combo_data.iloc[i,10].round(7)) + str(combo_data.iloc[i,11])
    if rec_id in rec_count:
        rec_count[rec_id] += 1
        if combo_data.iloc[i,20] == 1.0:
            rec_pos[rec_id] += 1
    else:
        rec_count[rec_id] = 1
        if combo_data.iloc[i,20] == 1.0:
            rec_pos[rec_id] = 1
        else:
            rec_pos[rec_id] = 0

for j in range(len(combo_data)):
    rec_id = str(combo_data.iloc[j,0].round(7)) + str(combo_data.iloc[j,1]) + str(combo_data.iloc[j,2]) + str(combo_data.iloc[j,3]) + str(combo_data.iloc[j,4]) + str(combo_data.iloc[j,5]) + str(combo_data.iloc[j,6]) + str(combo_data.iloc[j,7]) + str(combo_data.iloc[j,8].round(7)) + str(combo_data.iloc[j,9].round(7)) + str(combo_data.iloc[j,10].round(7)) + str(combo_data.iloc[j,11])
    combo_data.iloc[j,30] = rec_pos[rec_id]/rec_count[rec_id]

combo_data.to_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_gauss_aged_LIKECORRECTED_normalized_combo_nn_data.tsv', sep='\t', encoding='utf-8', index=False)