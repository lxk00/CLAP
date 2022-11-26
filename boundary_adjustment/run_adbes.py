from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from loss import *
import json
from sklearn.metrics import classification_report

class ModelManager:
    def __init__(self, args, data, pretrained_model=None):

        self.model = pretrained_model

        if self.model is None:
            self.model = BertForModel.from_pretrained(
                args.bert_model,
                cache_dir="",
                num_labels=data.num_labels,
                cosine=args.cosine,
                norm_output=args.do_bert_output_norm,
            )
            self.restore_model(args)
        self.model.num_neg = args.n
        self.trained = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_eval_score = 0
        self.best_ep = -1 

        self.delta = None
        self.delta_points = []
        self.centroids = None
        if os.path.exists(
            os.path.join(args.save_results_path, f"centroids{args.n}.npy")
        ):
            if not args.train_from_scratch:
                print("centroids and delta exists, loading")
                self.trained = True
                self.delta = torch.tensor(
                    np.load(os.path.join(args.save_results_path, f"deltas{args.n}.npy"))
                ).to(self.device)
                self.centroids = torch.tensor(
                    np.load(
                        os.path.join(args.save_results_path, f"centroids{args.n}.npy")
                    )
                ).to(self.device)
            else:
                print("centroids and delta exists, ignored and retrain")
        elif os.path.exists(
            os.path.join(args.pretrain_dir, f"best_centroids.pt")
        ):

            if not args.train_from_scratch:
                print("centroids and delta exists, loading")
                self.trained = True
                self.delta = torch.tensor(
                    torch.load(os.path.join(args.pretrain_dir, f"best_deltas.pt"))
                ).to(self.device)
                self.centroids = torch.tensor(
                    torch.load(
                        os.path.join(args.pretrain_dir, f"best_centroids.pt")
                    )
                ).to(self.device)

            else:
                print("centroids and delta exists, ignored and retrain")

        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.write_files = args.write_results
        self.cosine = args.cosine

        if self.write_files:
            self.save_result_tool = dict()

    def open_classify(self, features, others=False):

        logits = euclidean_metric(features, self.centroids, cosine=args.cosine)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        if self.cosine:
            euc_dis = 1 - torch.nn.functional.cosine_similarity(
                features, self.centroids[preds], dim=1
            )
        else:
            euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        new_delta = torch.cat([self.delta, torch.tensor([4], dtype=self.delta.dtype, device=self.delta.device)], dim=0)
        preds[euc_dis >= new_delta[preds]] = data.unseen_token_id
        if others:
            preds[preds == self.delta.shape[0]] = data.unseen_token_id

        if self.write_files:
            if self.cosine:
                return preds, 1 - logits
            return preds, -logits
        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        if mode == "eval":
            dataloader = data.eval_dataloader
        elif mode == "test":
            dataloader = data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration", mininterval=60):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(pooled_output, args.aug_gen)
                if self.write_files:
                    preds, logits = preds
                    for _logits, _label_ids in zip(logits, label_ids):
                        self.save_result_tool[len(self.save_result_tool)] = (
                            _logits.detach().cpu().tolist(),
                            _label_ids.detach().cpu().tolist(),
                        )
                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == "eval":
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)["F1-score"]
            return eval_score

        elif mode == "test":
            # print(classification_report(y_true, y_pred, digits=4, target_names=data.label_list))
            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results["Accuracy"] = acc

            self.test_results = results
            self.save_results(args)

    def train(self, args, data):
        if self.trained:
            print("BOUNDARY TRAINED")
            return
        criterion_boundary = BoundaryLoss(
            num_labels=data.num_labels,
            feat_dim=args.feat_dim,
            num_neg=args.n,
            safe1=args.safe1,
            safe2=args.safe2,
            alpha=args.alpha,
            cosine=args.cosine,
            norm=args.do_bert_output_norm,
        )

        if not self.cosine or not args.do_bert_output_norm:
            self.delta = F.softplus(criterion_boundary.delta)
        else:
            self.delta = criterion_boundary.delta

        optimizer = torch.optim.Adam(
            criterion_boundary.parameters(), lr=args.lr_boundary
        )
        
        self.centroids = self.centroids_cal(args, data)
        # self.centroids = torch.nn.functional.normalize(self.model.classifier.T)
        # self.centroids.requires_grad = False
        # self.centroids = torch.nn.functional.normalize(self.centroids, dim=1)
        ######################
        # self.centroids, new_delta = self.centroids_cal(args, data, True)
        # new_delta = torch.tensor(new_delta, dtype=criterion_boundary.delta.data.dtype, device=criterion_boundary.delta.data.device)
        # criterion_boundary.delta.data = new_delta
        # self.delta = new_delta

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(
                tqdm(data.train_dataloader, desc="Iteration", mininterval=60)
            ):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(
                        input_ids,
                        segment_ids,
                        input_mask,
                        feature_ext=True,
                        return_neg=True,
                    )
                    loss, self.delta = criterion_boundary(
                        features, self.centroids, label_ids[:: 1 + args.n]
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tr_loss += loss.item()

                    nb_tr_examples += input_ids.size(0) / (1 + args.n)
                    nb_tr_steps += 1

            # self.delta_points.append(self.delta)

            # if epoch <= 20:
            #     plot_curve(self.delta_points)

            loss = tr_loss / nb_tr_steps
            print("train_loss", loss)

            eval_score = self.evaluation(args, data, mode="eval")
            print("eval_score", eval_score)

    
            print(eval_score, torch.mean(self.delta).detach().cpu().numpy())

            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_ep = epoch
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.delta = best_delta
        self.centroids = best_centroids

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def delta_init_new(self, features, centroids, labels, cosine=False):
        if cosine:
            distances = 1 - features @ centroids.T  # (N, C)
        else:
            distances = euclidean_metric(features, centroids)
        deltas = []
        for i in range(distances.shape[1]):
            d_true = distances[:, i][labels == i]
            d_false = distances[:, i][labels != i]
            deltas.append((torch.mean(d_true) * 0.5 + torch.mean(d_false) * 0.5).item())
        return torch.tensor(deltas)

    def centroids_cal(self, args, data):
        if args.aug_gen:
            centroids = torch.zeros(data.num_labels + 1, args.feat_dim).cuda()
        else:
            centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        with torch.set_grad_enabled(False):
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                label_ids = label_ids[:: 1 + args.n]
                input_ids = input_ids[:: 1 + args.n]
                input_mask = input_mask[:: 1 + args.n]
                segment_ids = segment_ids[:: 1 + args.n]
                features = self.model(
                    input_ids, segment_ids, input_mask, feature_ext=True
                )
                # if new_delta:
                    # total_features.append(features)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
            if args.aug_gen:
                for batch in data.aug_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    label_ids = label_ids[:: 1 + args.n]
                    input_ids = input_ids[:: 1 + args.n]
                    input_mask = input_mask[:: 1 + args.n]
                    segment_ids = segment_ids[:: 1 + args.n]
                    features = self.model(
                        input_ids, segment_ids, input_mask, feature_ext=True
                    )
                    total_labels = torch.cat((total_labels, label_ids))
                    for i in range(len(label_ids)):
                        label = label_ids[i]
                        centroids[label] += features[i]

        total_labels = total_labels.cpu().numpy()
        centroids /= (
            torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        )
        if args.cosine or args.do_bert_output_norm:
            centroids = torch.nn.functional.normalize(centroids, dim=1)

        return centroids

    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        vars_dict = {
            "labeled_ratio": args.labeled_ratio,
            "dataset": args.dataset,
            "known_cls_ratio": args.known_cls_ratio,
            "seed": args.seed,
            "best_ep":self.best_ep
        }
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        np.save(
            os.path.join(args.save_results_path, f"centroids{args.n}.npy"),
            self.centroids.detach().cpu().numpy(),
        )
        np.save(
            os.path.join(args.save_results_path, f"deltas{args.n}.npy"),
            self.delta.detach().cpu().numpy(),
        )
        if self.write_files:
            fw = open(os.path.join(args.save_results_path, "distances.json"), "w")
            for k, v in self.save_result_tool.items():
                dist, label = v
                fw.write(
                    str(label)
                    + "\t"
                    + "\t".join([str(round(x, 4)) for x in dist])
                    + "\n"
                )

        file_name = "results.csv"
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print("test_results", data_diagram)


if __name__ == "__main__":

    print("Data and Parameters Initialization...")
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    print("Pre-training begin...")
    manager_p = PretrainModelManager(args, data)
    # r = manager_p.eval(args, data)
    manager_p.train(args, data)
    print("Pre-training finished!")

    manager = ModelManager(args, data, manager_p.model)
    print("Training begin...")
    manager.train(args, data)
    print("Training finished!")

    print("Evaluation begin...")
    manager.evaluation(args, data, mode="test")
    print("Evaluation finished!")

    # debug(data, manager_p, manager, args)

