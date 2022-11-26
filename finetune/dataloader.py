

import torch
import json_lines
from util import *
from collections import Counter
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:

    def __init__(self, args):
        set_seed(args.seed_data)
        max_seq_lengths = {'oos': 30, 'stackoverflow': 45, 'banking': 55}
        args.max_seq_length = max_seq_lengths[args.dataset]

        neg_num_dict = {'oos': 100, 'stackoverflow': 600, 'banking': 100}
        self.neg_num = neg_num_dict[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.num_labels = len(self.known_label_list)
        self.unknown_label_list = list(set(self.all_label_list)^set(self.known_label_list))
        print('len: {}, all_label_list: {} \n'.format(len(self.all_label_list), self.all_label_list))
        print('len: {}, known_label_list: {} \n'.format(len(self.known_label_list), self.known_label_list))
        print('len: {}, unknown_label_list: {} \n'.format(len(self.unknown_label_list), self.unknown_label_list))

        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'

        self.unseen_token_id = self.num_labels
        self.label_list = list(self.known_label_list) + [self.unseen_token]
        self.train_labels = list(self.known_label_list)
        print('len: {}, label_list: {}'.format(len(self.label_list), self.label_list))
        self.label_map = {}
        for i, label in enumerate(self.label_list):
            self.label_map[label] = i
        print('\n ==== self.label_map: {} ==== \n'.format(self.label_map))

        self.train_examples = self.get_examples(processor, args, 'train')   
        self.eval_examples = self.get_examples(processor, args, 'eval')     
        self.test_examples = self.get_examples(processor, args, 'test')     
        print('train: {}, eval: {}, test: {}'.format(len(self.train_examples), len(self.eval_examples), len(self.test_examples)))
        self.neg_gen_examples = None

        self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
        self.k_positive_dataloader = self.get_loader(self.train_examples, args, 'k_positive', self.neg_gen_examples)

    def read_jsonl(self, args):
        examples = []
        i = 1
        path = os.path.join(args.data_dir, '{}_v3.jsonl'.format(args.dataset))
        with open(path, "r", encoding="utf-8") as f:
            for item in json_lines.reader(f):
                for text in item['generate_other']:
                    if len(text) > 0:
                        guid = "%s-%s" % (args.adbes_type, i)
                        text_a = text
                        label = self.gen_label
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                        i += 1
        print('{}, len: {}'.format(path, len(examples)))
        return examples

    def get_examples(self, processor, args, mode='train'):
        mode_ = mode
        ori_examples = processor.get_examples(self.data_dir, mode_)

        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)
        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.known_label_list):
                    examples.append(example)
        elif mode == 'test':
            for example in ori_examples:
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    examples.append(example)
        print('{}, {}, before: {}, after: {} \n'.format(self.data_dir, mode, len(ori_examples), len(examples)))
        return examples

    def get_loader(self, examples, args, mode='train', neg_gen_examples=None):
        tokenizer = BertTokenizer.from_pretrained(args.bert_base_model, do_lower_case=True)
        if mode == 'k_positive':
            examples += neg_gen_examples
        features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        
        datatensor = TextDataset(input_ids, input_mask, segment_ids, label_ids, args.kccl_k, args.neg_num)

        if mode == 'train':
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.train_batch_size)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.eval_batch_size)
        elif mode == 'k_positive':
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.train_batch_size)
        print('mode: {}, len: {}'.format(mode, len(datatensor)))
        return dataloader

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, k_pos=0, n_neg=0,
                 mode='train', neg_label=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label2sid = dict()    
        self.k = k_pos
        self.n = n_neg
        self.mode = mode
        self.neg_label = neg_label

        if k_pos > 0:
            for sid, i in enumerate(label_ids.detach().cpu().numpy()):
                if i not in self.label2sid:
                    self.label2sid[i] = [sid]
                else:
                    self.label2sid[i].append(sid)

    def generate_postive_sample(self, label, self_index):
        if self.k > 0:
            index_list = [ind for ind in self.label2sid[label] if ind != self_index]
            return np.random.choice(index_list, size=self.k, replace=False)
        else:
            return None

    def generate_negtive_sample(self, label):
        if self.n > 0:
            index_list = []
            for key, value in self.label2sid.items():
                if key != label:
                    index_list += value
            return np.random.choice(index_list, size=self.n, replace=False)
        else:
            return None

    def __getitem__(self, idx):
        sid = self.generate_postive_sample(self.label_ids[idx].item(), idx)
        if self.n > 0:
            nid = self.generate_negtive_sample(self.label_ids[idx].item())
            sids = np.append([idx], sid)
            sids = np.append(sids, nid)
        else:
            sids = np.append([idx], sid)
        input_ids = self.input_ids[sids]    
        input_mask = self.input_mask[sids]
        segment_ids = self.segment_ids[sids]
        label_ids = self.label_ids[sids]
        return {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_ids}

    def __len__(self):
        return len(self.label_ids)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'mytest':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "mytest75.tsv")), "mytest")

    def get_labels(self, data_dir):
        """See base class."""
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            labels.append(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        print('{}, len: {}, labels: {} \n'.format(set_type, len(dict(Counter(labels))), dict(Counter(labels))))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)            
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else: 
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  
        else:
            tokens_b.pop()
