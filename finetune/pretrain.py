from model import *
from dataloader import *
from util import *
from loss import *

from transformers import BertConfig
from sklearn.metrics import classification_report


class PretrainModelManager:
    def __init__(self, args, data):
        self.num_labels = data.num_labels
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_base_model, do_lower_case=True)  
        
        self.model = BertForModel.from_pretrained(args.bert_base_model, cache_dir="", num_labels=self.num_labels, model_type=args.model_type)

        if args.freeze_bert_parameters:
            for name, param in self.model.bert.named_parameters():
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        if 'odist' in args.adbes_type:
            self.num_train_optimization_steps = int((len(data.train_examples) + len(data.neg_gen_examples)) / args.train_batch_size) * args.num_pretrain_epochs
        else:
            self.num_train_optimization_steps = int(
                (len(data.train_examples) / args.train_batch_size)) * args.num_pretrain_epochs
            
        print('self.num_train_optimization_steps: ', self.num_train_optimization_steps)
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 0

    def eval(self, args, data, epoch=0):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Pre-Train Eval Iteration", mininterval=10):
        
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode='eval', args=args)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        try:
            classfication = classification_report(y_true, y_pred, digits=4, target_names=data.train_labels)
        except Exception as e:
            print('error: ', e)
            classfication = classification_report(y_true, y_pred, digits=4)
        return acc, classfication

    def train(self, args, data):
        wait = 0
        best_model = None

        for epoch in trange(int(args.num_pretrain_epochs), desc="Pre-Train Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.centroids = None
            
            for step, batch in enumerate(tqdm(data.k_positive_dataloader, desc="Pre-Train Iteration", mininterval=10)):
                input_ids = batch['input_ids'].view(-1, args.max_seq_length).to(self.device)    
                input_mask = batch['input_mask'].view(-1, args.max_seq_length).to(self.device)
                segment_ids = batch['segment_ids'].view(-1, args.max_seq_length).to(self.device)
                label_ids = batch['label_ids'].view(-1).to(self.device)     
                
                with torch.set_grad_enabled(True):
                    loss, pooled_output, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                             mode="train", args=args, centroids=self.centroids)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    

                    tr_loss += loss.item()
                    
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            eval_score, classfication = self.eval(args, data, epoch)
            print('epoch: {}, loss: {}, eval_score_acc: {}'.format(epoch, loss, eval_score))

            if eval_score >= self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
                
                self.save_model(args, self.model, self.tokenizer)
                print(classfication)
                print('save the best pretrained models, seed: {}, epoch: {}, eval_score_acc: {}, save dir: {}'.format(args.seed, epoch, eval_score, args.pretrain_model_path))
            else:
                wait += 1
                if wait >= args.wait_patient:
                    print('wait={} >= args.wait_patient={}, break!'.format(wait, args.wait_patient))
                    break
            if args.le_random == 0:
                centroids = self.centroids_cal(args, data)
                self.model.weight = torch.nn.Parameter(centroids)
                print('self.model.weight updates: {}-mean'.format(args.le_random))
        self.model = best_model
        

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.pretrain_lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    def save_pretrain_model(self, args, model, tokenizer):
        
        model.save_pretrained(args.pretrain_model_path)
        tokenizer.save_pretrained(args.pretrain_model_path)

    def save_model(self, args, best_model, tokenizer):
        
        save_model = best_model.module if hasattr(best_model, 'module') else best_model
        model_file = os.path.join(args.pretrain_model_path, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_model_path, CONFIG_NAME)
        
        torch.save(save_model.state_dict(), model_file)
        save_model.config.to_json_file(model_config_file)
        tokenizer.save_vocabulary(args.pretrain_model_path)
        torch.save(self.best_eval_score, os.path.join(args.pretrain_model_path, 'best_pretrained_acc.pt'))

    def reload_model(self, args):
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
        model = BertForModel.from_pretrained(args.pretrain_model_path, cache_dir="", num_labels=self.num_labels, model_type=args.model_type)
        output_model_file = os.path.join(args.pretrain_model_path, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        try:
            best_pretrained_acc = torch.load(os.path.join(args.pretrain_model_path, 'best_pretrained_acc.pt'))
        except Exception as e:
            print(e)
            best_pretrained_acc = 0
        return model, tokenizer, best_pretrained_acc

    def restore_model(self, args):
        model = BertForModel.from_pretrained(args.pretrain_model_path, self.num_labels, model_type=args.model_type)
        output_model_file = os.path.join(args.pretrain_model_path, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
        return model, tokenizer

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
            for batch in data.k_positive_dataloader:
                input_ids = batch['input_ids'].view(-1, args.max_seq_length).to(self.device)
                input_mask = batch['input_mask'].view(-1, args.max_seq_length).to(self.device)
                segment_ids = batch['segment_ids'].view(-1, args.max_seq_length).to(self.device)
                label_ids = batch['label_ids'].view(-1).to(self.device)
                features = self.model(input_ids, segment_ids, input_mask, label_ids, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]

        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        return centroids
