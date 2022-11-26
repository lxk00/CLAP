from util import *
import torch.nn.functional as F



class BertForModel(BertPreTrainedModel):
    def __init__(self,config,num_labels,num_neg=0, cosine=False, norm_output=False):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.cosine = cosine

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.cosine:
            self.classifier = nn.Parameter(torch.rand(config.hidden_size, num_labels))
        else:
            self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.num_neg = num_neg
        self.norm_output = norm_output
        self.layer_mean = lambda output, mask: torch.sum(output * mask.unsqueeze(-1).repeat(1, 1, output.shape[-1]), dim=1) / torch.sum(mask, dim=1, keepdim=True)


    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None, return_neg=False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        encoded_layer_12 = encoded_layer_12[-1]
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        # pooled_output = self.dense(self.layer_mean(encoded_layer_12, attention_mask))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        if self.norm_output:
            pooled_output = torch.nn.functional.normalize(pooled_output, dim=1)

        if self.cosine:
            cls_tensor = torch.nn.functional.normalize(self.classifier, dim=0)
            pooled_output = torch.nn.functional.normalize(pooled_output, dim=-1)
            logits = pooled_output @ cls_tensor
        else:
            logits = self.classifier(pooled_output)
        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = nn.CrossEntropyLoss()(logits,labels)
                return loss
            else:
                return pooled_output, logits

