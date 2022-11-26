from util import *


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def cosine_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = torch.cosine_similarity(a, b, dim=2)
    return logits


def cosine_distance(a, b, dim=1):
    cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
    output = cos(a, b)
    return output


class BoundaryLoss(nn.Module):
    def __init__(self, num_labels=10, feat_dim=768, train_batch_size=512):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        # self.delta = nn.Parameter((torch.randint(2, 7, (num_labels, 1)) / 10).squeeze(-1).cuda())   # [0.2, 0.7]
        nn.init.normal_(self.delta)

    def forward(self, pooled_output, centroids, labels, centroids_norm=0, metric_type=0, softplus=1, poolout_norm=0):
        pooled_output = F.normalize(pooled_output)
        centroids = F.normalize(centroids)
        logits = euclidean_metric(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

        if softplus == 1:
            delta = F.softplus(self.delta)
        else:
            delta = self.delta
        c = centroids[labels]   
        d = delta[labels]       
        x = pooled_output       
        if metric_type == 1:
            # cos_dis = torch.diag(1 - cosine_metric(x, c))
            cos_dis = 1 - cosine_distance(x, c)
            pos_mask = (cos_dis > d).type(torch.cuda.FloatTensor)
            neg_mask = (cos_dis < d).type(torch.cuda.FloatTensor)
            pos_loss = (cos_dis - d) * pos_mask
            neg_loss = (d - cos_dis) * neg_mask
            # print(cos_dis, d)
            loss = pos_loss.mean() + neg_loss.mean()
        else:
            euc_dis = torch.norm(x - c, 2, 1).view(-1)
            pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
            neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
            pos_loss = (euc_dis - d) * pos_mask
            neg_loss = (d - euc_dis) * neg_mask
            loss = pos_loss.mean() + neg_loss.mean()
        return loss, delta
