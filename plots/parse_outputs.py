import pandas as pd

method_names = ['ABE0', 'ATLM', 'CART', 'CoGEE', 'MOEAD', 'DE30', 'GA100', 'DE10', 'NSGA2']


def reading(model_index, model_name):

    if type(model_index) is int:
        model_index = str(model_index)

    data = pd.read_csv('Outputs/final_list_2.txt', sep=";", header=None)
    # data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG", "NGEN"]
    data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "Runtime"]

    mre_file_name = model_name + '_mre.txt'
    sa_file_name = model_name + '_sa.txt'

    for methodid in [1, 2, 3]:
        df = data.query('Data_ID == ["' + model_index + '"] and Method_ID == ["' + str(methodid) + '"]')
        mre_str = ' '.join([str(i) for i in sorted(df.loc[:, "MRE"])])
        sa_str = ' '.join([str(i) for i in sorted(df.loc[:, "SA"])])

        with open(mre_file_name, 'a+') as f:
            f.write('\n\n' + method_names[methodid] + '\n' + mre_str)

        with open(sa_file_name, 'a+') as f:
            f.write('\n\n' + method_names[methodid] + '\n' + sa_str)


if __name__ == '__main__':
    reading('0', 'albrecht')