import os
import io
import pandas as pd
import torch
from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from loss import *
from sklearn.metrics import classification_report

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        
        self.log.write(message)

    def flush(self):
        pass


class ModelManager:

    def __init__(self, args, data, pretrained_model=None, pretrained_tokenizer=None, pretrained_acc=0):

        self.pretrained_acc = pretrained_acc
        self.model = pretrained_model
        self.tokenizer = pretrained_tokenizer

        if self.model is None:
            self.model = BertForModel.from_pretrained(args.pretrain_model_path, cache_dir="", num_labels=data.num_labels, model_type=args.model_type)
            
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)  

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_eval_score = 0
        self.best_eval_score_acc = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def cos_cal(self, features):
        if args.metric_type == 1:
            logits = cosine_metric(features, self.centroids)
            probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            cos_dis = torch.diag(1 - cosine_metric(features, self.centroids[preds]))
            return cos_dis, self.delta[preds], preds
        else:
            logits = euclidean_metric(features, self.centroids)
            probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
            return euc_dis, self.delta[preds], preds

    def open_classify(self, features):
        if args.metric_type == 1:
            logits = cosine_metric(features, self.centroids)
            
            probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            cos_dis = torch.diag(1 - cosine_metric(features, self.centroids[preds]))
            
            
            preds[cos_dis >= self.delta[preds]] = data.unseen_token_id
        else:
            logits = euclidean_metric(features, self.centroids)
            probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
            
            preds[euc_dis >= self.delta[preds]] = data.unseen_token_id
        return preds


    def save_adbes_model(self, args, model, tokenizer):
        save_model = model.module if hasattr(model, 'module') else model
        model_file = os.path.join(args.adbes_model_path, WEIGHTS_NAME)
        model_config_file = os.path.join(args.adbes_model_path, CONFIG_NAME)
        torch.save(save_model.state_dict(), model_file)
        save_model.config.to_json_file(model_config_file)
        tokenizer.save_vocabulary(args.adbes_model_path)

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds_ = torch.empty(0, dtype=torch.long).to(self.device)
        total_cos_dis = torch.empty(0, dtype=torch.long).to(self.device)
        total_delta_preds = torch.empty(0, dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader
            self.delta = torch.load(os.path.join(args.adbes_model_path, 'best_deltas.pt'))
            self.centroids = torch.load(os.path.join(args.adbes_model_path, 'best_centroids.pt'))
            print('load the best delta and centroids from path: {}'.format(args.adbes_model_path))
        else:
            raise Exception('the mode of evaluation is not required.')

        for batch in tqdm(dataloader, desc="Iteration", mininterval=10):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(pooled_output)
                cos_dis, delta_preds, preds_ = self.cos_cal(pooled_output)
                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))
                total_preds_ = torch.cat((total_preds_, preds_))
                total_cos_dis = torch.cat((total_cos_dis, cos_dis))
                total_delta_preds = torch.cat((total_delta_preds, delta_preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_pred_ = total_preds_.cpu().numpy()
        cos_distance = total_cos_dis.cpu().numpy().tolist()
        cos_delta = total_delta_preds.cpu().numpy().tolist()

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['all_f1']
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            try:
                classification = classification_report(y_true, y_pred, digits=4, target_names=data.label_list)
            except Exception as e:
                print('error: {}'.format(e))
                classification = classification_report(y_true, y_pred, digits=4)
            print(acc)
            
            return eval_score, acc, classification

        elif mode == 'test':
            try:
                print(classification_report(y_true, y_pred, digits=4, target_names=data.label_list))
            except Exception as e:
                print('error: {}'.format(e))
                print(classification_report(y_true, y_pred, digits=4))

            text_list = [example.text_a for example in data.test_examples]
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            predictions_ = list([data.label_list[idx] for idx in y_pred_])
            self.true_labels = list([data.label_list[idx] for idx in y_true])
            
            def flag_mode(true_labels, predictions, data):
                flag = []
                for idx in range(len(true_labels)):
                    if true_labels[idx] == data.gen_label:
                        if predictions[idx] in [data.gen_label, data.unseen_token]:
                            flag_tmp = True
                        else:
                            flag_tmp = False
                    else:
                        if true_labels[idx] == predictions[idx]:
                            flag_tmp = True
                        else:
                            flag_tmp = False
                    flag.append(flag_tmp)
                return flag

            flag = flag_mode(self.true_labels, self.predictions, data)

            df = pd.DataFrame(
                {'text': text_list, 'true': self.true_labels, 'pred': self.predictions, 'pred_': predictions_, 'distance': cos_distance, 'delta': cos_delta, 'flag': flag})
            outpath = os.path.join(args.adbes_model_path, 'pred_{}.xlsx'.format(args.metric_type))
            df.to_excel(outpath, index=False, header=True)
            print('save to {}'.format(outpath))

            known_acc, other_acc, unknown_acc, all_acc, other_unk, all_mod_acc = [], [], [], [], [], []
            other2unk, unk2other = [], []
            for fi in range(len(self.predictions)):
                all_acc.append(1 if self.true_labels[fi] == self.predictions[fi] else 0)
                if self.true_labels[fi] != data.unseen_token:
                    if self.true_labels[fi] == data.gen_label:
                        other_acc.append(1 if self.true_labels[fi] == self.predictions[fi] else 0)
                    else:
                        known_acc.append(1 if self.true_labels[fi] == self.predictions[fi] else 0)
                else:
                    unknown_acc.append(1 if self.true_labels[fi] == self.predictions[fi] else 0)
                if self.true_labels[fi] in [data.gen_label, data.unseen_token]:
                    if self.true_labels[fi] == data.gen_label:
                        other2unk.append(1 if self.predictions[fi] == data.unseen_token else 0)
                    else:
                        unk2other.append(1 if self.predictions[fi] == data.gen_label else 0)
                    self.true_labels[fi] = 'OTHER_UNK'
                    if self.predictions[fi] in [data.gen_label, data.unseen_token]:
                        self.predictions[fi] = 'OTHER_UNK'
                        other_unk.append(1)
                    else:
                        other_unk.append(0)
                if self.predictions[fi] in [data.gen_label, data.unseen_token]:
                    self.predictions[fi] = 'OTHER_UNK'
                all_mod_acc.append(1 if self.true_labels[fi] == self.predictions[fi] else 0)
            print('acc: {} / {} = {}'.format(sum(all_acc), len(all_acc), round(sum(all_acc) / len(all_acc), 4)))
            print('acc_mod: {} / {} = {}'.format(sum(all_mod_acc), len(all_mod_acc), round(sum(all_mod_acc) / len(all_mod_acc), 4)))
            print('known_acc: {} / {} = {}'.format(sum(known_acc), len(known_acc), round(sum(known_acc) / len(known_acc), 4)))

            df = pd.DataFrame(
                {'text': text_list, 'true': self.true_labels, 'pred': self.predictions, 'pred_': predictions_, 'distance': cos_distance, 'delta': cos_delta, 'flag': flag_mode(self.true_labels, self.predictions, data)})
            outpath = os.path.join(args.adbes_model_path, 'pred_mod.xlsx')
            df.to_excel(outpath, index=False, header=True)
            print('save to {}'.format(outpath))

            def result_dict(y_true, y_pred, known_acc, unknown_acc, mode=None):
                cm = confusion_matrix(y_true, y_pred)
                result = F_measure(cm)
                acc = round(accuracy_score(y_true, y_pred) * 100, 2)
                results = {}
                results['mode'] = mode
                results['dataset'] = args.dataset
                results['known_cls_ratio'] = args.known_cls_ratio
                results['seed'] = args.seed
                results['pretrain_acc'] = self.pretrained_acc
                results['all_acc'] = acc
                results['all_f1'] = result['all_f1']
                results['known_acc'] = round((sum(known_acc) / len(known_acc)) * 100, 2)
                results['open_acc'] = round((sum(unknown_acc) / len(unknown_acc)) * 100, 2)
                results['known_f1'] = result['known_f1']
                results['open_f1'] = result['open_f1']
                results['kccl_k'] = args.kccl_k
                results['temperature'] = args.temperature
                results['KCCL_LOSS_LAMBDA'] = args.KCCL_LOSS_LAMBDA
                results['CE_LOSS_LAMBDA'] = args.CE_LOSS_LAMBDA
                print(results)
                return results
            self.save_results(args, result_dict(y_true, y_pred, known_acc, unknown_acc, 'src'))
            self.save_results(args, result_dict(self.true_labels, self.predictions, known_acc, other_unk, 'modify'))


    def train(self, args, data):
        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim, train_batch_size=args.train_batch_size)
        self.delta = F.softplus(criterion_boundary.delta)
        if args.optimizer_lr == 1:
            total_steps = int(len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
            optimizer = BertAdam(criterion_boundary.parameters(),
                                 lr=args.lr_boundary,
                                 warmup=args.warmup_proportion,
                                 t_total=total_steps)
        else:
            optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr=args.lr_boundary)
        if args.centroids == 0:
            self.centroids = self.centroids_cal(args, data)
            print('use self.centroids_cal as centroids: {}'.format(self.centroids.shape))
        elif args.centroids == 1:
            self.centroids = self.model.weight
            print('use self.model.weight as centroids: {}'.format(self.centroids.shape))
            torch.save(self.centroids, os.path.join(args.adbes_model_path, 'model_weight_src.pt'))

        wait = 0
        best_delta, best_centroids = None, None
        best_model = None

        for epoch in trange(int(args.num_train_epochs), desc="Train Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            

            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Train Iteration", mininterval=10)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids, args.centroids_norm, args.metric_type, args.softplus, args.poolout_norm)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                

            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps
            eval_score, acc, classification = self.evaluation(args, data, mode="eval")
            print('epoch: {}, loss: {}, eval_score_f1: {}, eval_score_acc: {}'.format(epoch, loss, eval_score, acc))


            if args.eval_metric == 'f1':
                
                if eval_score >= self.best_eval_score:
                    wait = 0
                    best_model = copy.deepcopy(self.model)
                    self.best_eval_score = eval_score
                    best_delta = self.delta
                    best_centroids = self.centroids
                    self.save_adbes_model(args, self.model, self.tokenizer)
                    print(classification)
                    print('save the best adbes models epoch: {}, save dir: {}'.format(epoch, args.adbes_model_path))
                    torch.save(best_centroids, os.path.join(args.adbes_model_path, 'best_centroids.pt'))
                    torch.save(best_delta, os.path.join(args.adbes_model_path, 'best_deltas.pt'))
                    print('finish to save best_deltas.npy and best_centroids.npy to: {}'.format(args.adbes_model_path))
                else:
                    wait += 1
                    if wait >= args.wait_patient:
                        print('wait={} >= args.wait_patient={}, break!'.format(wait, args.wait_patient))
                        break
            else:
                
                if acc >= self.best_eval_score_acc:
                    wait = 0
                    best_model = copy.deepcopy(self.model)
                    self.best_eval_score_acc = acc
                    best_delta = self.delta
                    best_centroids = self.centroids
                    
                    self.save_adbes_model(args, self.model, self.tokenizer)
                    print(classification)
                    print('save the best adbes models epoch: {}, save dir: {}'.format(epoch, args.adbes_model_path))
                    torch.save(best_centroids, os.path.join(args.adbes_model_path, 'best_centroids.pt'))
                    torch.save(best_delta, os.path.join(args.adbes_model_path, 'best_deltas.pt'))
                    print('finish to save best_deltas.npy and best_centroids.npy to: {}'.format(args.adbes_model_path))
                else:
                    wait += 1
                    if wait >= args.wait_patient:
                        print('wait={} >= args.wait_patient={}, break!'.format(wait, args.wait_patient))
                        break

        self.delta = best_delta
        self.centroids = best_centroids
        self.model = best_model

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            for batch in data.odist_train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]

        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        return centroids

    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_model_path, 'pytorch_model.bin')
        self.model.load_state_dict(torch.load(output_model_file))

    def save_results(self, args, results):
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results.csv'
        results_path = os.path.join(args.adbes_model_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, sep=',', index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, sep=',', index=False)
        data_diagram = pd.read_csv(results_path)
        print(results_path)
        pd.set_option('display.max_columns', None)  
        pd.set_option('display.max_rows', None)     
        pd.set_option('display.width', 100)         
        print(data_diagram)


def reload_adbes_model(args):
    config = BertConfig.from_json_file(os.path.join(args.adbes_model_path, 'config.json'))
    tokenizer = BertTokenizer.from_pretrained(args.adbes_model_path)
    
    model = BertForModel.from_pretrained(args.adbes_model_path, cache_dir="", num_labels=data.num_labels, model_type=args.model_type)
    output_model_file = os.path.join(args.adbes_model_path, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model, tokenizer


def create_model_dir(args, mode='pretrain', path=''):
    model_name = 'KCCL'
    if mode == 'pretrain':
        model_path = os.path.join(
            args.pretrain_dir, args.dataset, args.dataset_mode + '_' + model_name + '_' + str(args.known_cls_ratio), str(args.save_version))
    elif mode == 'train':
        model_path = os.path.join(
            args.save_results_dir, args.dataset, args.dataset_mode + '_' + model_name + '_' + str(args.known_cls_ratio), str(args.save_version))
    else:
        raise Exception('the mode of create_model_dir is not required.')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


if __name__ == '__main__':
    start_time = datetime.now()

    print(datetime.now(), ' Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    args.save_version = '{}{}_m{}_l{}_d{}_p{}_te{}_t{}_{}_s{}_sd{}_{}_NEG'.format(args.dataset[0], str(args.known_cls_ratio), str(args.model_type), str(args.pretrain_loss_type), str(args.metric_type), str(args.kccl_k), str(args.temperature), str(args.KCCL_LOSS_LAMBDA), str(args.CE_LOSS_LAMBDA), str(args.seed), str(args.seed_data), args.adbes_type)
    args.pretrain_model_path = create_model_dir(args, 'pretrain')
    if args.save_path_suffix == '':
        args.adbes_model_path = create_model_dir(args, 'train')
    else:
        args.adbes_model_path = create_model_dir(args, 'train_nopre', path=args.save_path_suffix)

    sys.stdout = Logger('train.log', path=args.adbes_model_path)

    print('start_time:{}'.format(start_time))
    print('args: ', args)

    data = Data(args)
    print(datetime.now(), ' Data and Parameters finished loading!')

    set_seed_all(args.seed)
    if not os.path.exists(os.path.join(args.pretrain_model_path, 'pytorch_model.bin')):
        print(datetime.now(), ' Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print(datetime.now(), ' Pre-training finished!')

    print(datetime.now(), ' Pre-training begin loading...')
    manager_p = PretrainModelManager(args, data)
    pretrained_model, pretrained_tokenizer, best_pretrained_acc = manager_p.reload_model(args)
    print(datetime.now(), ' Pre-training finished loading!')

    if not os.path.exists(os.path.join(args.adbes_model_path, 'pytorch_model.bin')):
        manager = ModelManager(args=args, data=data, pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer, pretrained_acc=best_pretrained_acc)
        print(datetime.now(), ' Training begin...')
        manager.train(args, data)
        print(datetime.now(), ' Training finished!')
    else:
        print(datetime.now(), ' Training begin loading...')
        train_model, train_tokenizer = reload_adbes_model(args)
        manager = ModelManager(args=args, data=data, pretrained_model=train_model, pretrained_tokenizer=train_tokenizer, pretrained_acc=best_pretrained_acc)
        print(datetime.now(), ' Training finished loading!')

    print(datetime.now(), ' Evaluation begin...')
    manager.evaluation(args, data, mode="test")
    print(datetime.now(), ' Evaluation finished!')
