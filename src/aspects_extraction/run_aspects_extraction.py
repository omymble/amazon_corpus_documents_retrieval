from ae_functions import *


def parse_absa_bert():
    targets_preprocessed = pd.read_pickle(DATA_PATH + 'targets_preprocessed.pkl')
    train_t5_base(targets_preprocessed)


def parse_absa_t5():
    return


def main(model_name='bert-base-cased'):
    if model_name == 'bert-base-cased':
        parse_absa_bert()
    if model_name == 't5-base-uncased':
        parse_absa_t5()


if __name__ == '__main__':
    main()
