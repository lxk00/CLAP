import argparse

def init_model():
    parser = argparse.ArgumentParser()

    # bert model
    parser.add_argument("--lang", default='english', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--bert_base_model",
                        default="pretrain_model/uncased_L-12_H-768_A-12",
                        type=str, help="The path for the pre-trained bert models.")

    # data
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="The name of the dataset to train selected")

    parser.add_argument("--dataset_mode", default='fixed', type=str, help="get dataset mode: fixed or random")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True,
                        help="The number of known classes")

    parser.add_argument("--labeled_ratio", default=1.0, type=float,
                        help="The ratio of labeled samples in the training set")

    # pretrain model
    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str,
                        help="The dir for the pre-trained and fine-tuned bert models.")

    parser.add_argument("--pretrain_model_path", default=None, type=str,
                        help="The path for the pre-trained and fine-tuned models.")

    parser.add_argument("--pretrain_loss_type", default=1, type=int)

    parser.add_argument("--kccl_k", default=1, type=int, help="The k value of KCCL")

    parser.add_argument("--neg_num", default=1, type=int, help="The num value of negative examples")

    parser.add_argument("--temperature", default=1.0, type=float, help="The temperature value of KCCL")

    parser.add_argument("--KCCL_LOSS_LAMBDA", default=1.0, type=float, help="The temperature value of KCCL")

    parser.add_argument("--CE_LOSS_LAMBDA", default=1.0, type=float, help="The temperature value of KCCL")

    parser.add_argument("--LMCL_LOSS_LAMBDA", default=1.0, type=float, help="The temperature value of LMCL")

    parser.add_argument("--metric_type", default=0, type=int, help="The metric_type of train: 0-euc, 1-cos")

    parser.add_argument("--kccl_euc", default=0, type=int, help="The method value of whether use euc or not in KCCL: 1-euc, 2-Contrastive Loss, 3-Triplet Loss")

    parser.add_argument("--ks", default=30, type=float, help="The s value of KCCL")

    parser.add_argument("--km", default=0.35, type=float, help="The margin value of KCCL")

    parser.add_argument("--s_v", default=30, type=float, help="The s value of LMCL")

    parser.add_argument("--m", default=0.35, type=float, help="The margin value of LMCL")

    parser.add_argument("--c_m", default=0.35, type=float, help="The margin value of Contrastive Loss")

    parser.add_argument("--t_a", default=0.35, type=float, help="The margin value of Triplet Loss")

    parser.add_argument("--neg_margin", default=0, type=int,
                        help="The method of whether add neg margin or not")

    parser.add_argument("--neg_m", default=0.35, type=float, help="The neg margin value of LMCL")

    parser.add_argument("--loss_metric", default=0, type=int,
                        help="The method of loss metric cal")

    parser.add_argument("--neg_method", default=0, type=int,
                        help="The method of neg example: ")

    parser.add_argument("--centroids", default=0, type=int,
                        help="The method of centroids")

    parser.add_argument("--centroids_norm", default=0, type=int,
                        help="The method of centroids_norm")

    parser.add_argument("--poolout_norm", default=0, type=int,
                        help="The method of poolout_norm")

    parser.add_argument("--model_type", default='0', type=str, help="The method of model")

    parser.add_argument("--adbes_type", default='train', type=str, help="The adbes method of model")

    parser.add_argument("--le_random", default=1, type=int, help="The method of le: 0-mean, 1-random")

    parser.add_argument("--optimizer_lr", default=0, type=int, help="The method of lr-boudary optimizer")

    parser.add_argument("--softplus", default=1, type=int, help="The method of whether use softplus or not")


    # train model
    parser.add_argument("--save_results_dir", type=str, default='outputs', help="the dir to save results")
    
    parser.add_argument("--adbes_model_path", default=None, type=str,
                        help="The adbes model path where the final models predictions and checkpoints will be written.")

    parser.add_argument('--save_version', type=str, default='0', help="the version of the adbes save model")

    parser.add_argument('--save_path_suffix', type=str, default='', help="the version suffix of the adbes save model")

    parser.add_argument('--eval_metric', type=str, default='acc', help="the eval_metric of the adbes save model: acc: accuracy, f1: f1-score")

    # model parameter
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument('--seed_data', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT")

    parser.add_argument("--num_pretrain_epochs", default=6.0, type=float, help="Total number of pretraining epochs to perform.")

    parser.add_argument("--num_train_epochs", default=6.0, type=float, help="Total number of training epochs to perform.")

    parser.add_argument("--pretrain_lr", default=2e-5, type=float, help="The learning rate of pretrian.")

    parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")

    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation.")
    
    parser.add_argument("--wait_patient", default=10, type=int, help="Patient steps for Early Stop.")

    return parser
