import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
import random
import collections
from torch.utils.data._utils.collate import default_collate

class ContrastiveDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels=None, p=0, n=0, neg_from_gen=False, guid=None, neg_examples=None, neg_guid=None):
        super().__init__()
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask=attention_mask
        self.labels = labels
        self.p = p
        self.n = n
        
        self.neg_from_gen = neg_from_gen
        

        if self.p + self.n != 0 and labels is not None:
            self.labels_index_dict = dict()
            if not self.neg_from_gen:
                for idx, i in enumerate(self.labels):
                    i = i.item()
                    if i not in self.labels_index_dict:
                        self.labels_index_dict[i] = [idx]
                    else:
                        self.labels_index_dict[i].append(idx)
            else:
                if neg_examples is None:
                    raise ValueError("ContrastiveDataset neg_examples can not be None")
                self.guid = [x.split('-')[-1] for x in guid]
                self.neg_guid = [x.split('-')[-1] for x in neg_guid]
                self.neg_examples = neg_examples
                self.neg_label_to_id = dict()
                for idx, label_i in enumerate([f.label_id for f in neg_examples]):
                    if label_i in self.neg_label_to_id:
                        self.neg_label_to_id[label_i].append(idx)
                    else:
                        self.neg_label_to_id[label_i] = [idx]
                self.neg_input_ids = torch.tensor([f.input_ids for f in neg_examples], dtype=torch.long)
                self.neg_input_mask = torch.tensor([f.input_mask for f in neg_examples], dtype=torch.long)
                self.neg_segment_ids = torch.tensor([f.segment_ids for f in neg_examples], dtype=torch.long)
                self.neg_label_ids = torch.tensor([f.label_id for f in neg_examples], dtype=torch.long)



    def get_pos(self, index):
        if self.p == 0:
            return []
        cur_label = self.labels[index].item()
        pairs = random.sample(self.labels_index_dict[cur_label], self.p)
        return pairs

    def get_neg(self, index):
        pairs = []
        cur_label = self.labels[index].item()
        for i in range(self.n):
            neg_label = random.choice(list(set(self.labels_index_dict.keys()) - set([cur_label])))
            pairs.append(random.choice(self.labels_index_dict[neg_label]))
        return pairs

    def get_neg_from_guid(self, guid, label):
        if guid in self.neg_guid:
            return self.neg_guid.index(guid)
        else:
            return random.choice(self.neg_label_to_id[label])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        if self.p + self.n != 0 and self.labels is not None:
            pos_index = self.get_pos(index)
            if not self.neg_from_gen:
                neg_index = self.get_neg(index)
                index = [index] + pos_index + neg_index
                a = self.input_ids[index, :]
                c = self.attention_mask[index, :]
                d = self.labels[index]
                b = self.token_type_ids[index, :]
            else:
                d = self.labels[index]
                neg_index = self.get_neg_from_guid('-1', d.item())
                neg_a = self.neg_input_ids[neg_index]
                neg_b = self.neg_segment_ids[neg_index]
                neg_c = self.neg_input_mask[neg_index]
                a = self.input_ids[index, :]
                c = self.attention_mask[index, :]
                d = self.labels[index]
                b = self.token_type_ids[index, :]
                a = torch.stack([a, neg_a])
                b = torch.stack([b, neg_b])
                c = torch.stack([c, neg_c])
                d = torch.stack([d, torch.max(self.labels)])

        else:
            a = self.input_ids[index]
            c = self.attention_mask[index]
            d = self.labels[index]
            b = self.token_type_ids[index]
        # return {"input_ids": a, "token_type_ids": b, "attention_mask":c, "labels":d}
        return [a, c, b, d]

    def collate_fn(self, batch):
        if type(batch) is not list:
            return default_collate(batch)
        elif type(batch[0][0]) is not torch.Tensor:
            return default_collate(batch)
        elif len(batch[0][0].shape) == 1:
            return default_collate(batch)
        else:
            b = len(batch)
            cat_batch = []
            for k in range(len(batch[0])):
                cat_batch.append(torch.cat([d[k] for d in batch], dim=0))
        return cat_batch

if __name__ == "__main__":
    input_ids = torch.arange(0, 10, dtype=torch.long).unsqueeze(1).repeat(1, 10)
    print(input_ids)
    print(input_ids.reshape(-1, 2, 10)[:, 0, :])
    print(input_ids.reshape(-1, 2, 10)[:, 1, :])
    exit(0)
    
    attention_mask = torch.zeros(10, 10, dtype=torch.long)
    token_type_ids = torch.zeros(10, 10, dtype=torch.long)
    labels = torch.tensor([0,1,2,0,1,2,0,1,2,0], dtype=torch.long)


    # datatensor = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    datatensor = ContrastiveDataset(input_ids, attention_mask, token_type_ids, labels, n=2)
    sampler = RandomSampler(datatensor)
    dataloader = DataLoader(datatensor, sampler=sampler, batch_size = 2, collate_fn=datatensor.collate_fn)
    # dataloader = DataLoader(datatensor, sampler=sampler, batch_size = 2)#, collate_fn=datatensor.collate_fn)
    for i in dataloader:
        print(i)    
        exit(0)