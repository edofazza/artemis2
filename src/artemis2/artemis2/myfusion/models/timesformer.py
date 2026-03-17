import csv
import math
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import TimesformerModel, logging


class TimeSformer(nn.Module):
    def __init__(self, num_frames):
        super(TimeSformer, self).__init__()
        self.num_classes = 140
        # BRANCH 1
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                         num_frames=num_frames, ignore_mismatched_sizes=True) #, local_files_only=True)

        # ACTION RECOGNITION
        self.group_linear = GroupWiseLinear(self.num_classes, self.backbone.config.hidden_size, bias=True)

    def forward(self, images):
        x = self.backbone(images)[0]
        out = self.group_linear(F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_classes).transpose(1, 2))
        #print(out.shape)
        return out


class TimeSformerExecutor(object):
    def __init__(self, test_loader, criterion, eval_metric, class_list, gpu_id) -> None:
        super().__init__()
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        self.class_list = class_list
        self.gpu_id = gpu_id
        num_frames = self.test_loader.dataset[0][0].shape[0]
        logging.set_verbosity_error()
        model = TimeSformer(num_frames).to(gpu_id)
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}])
        #self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt

    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = (data.to(self.gpu_id), label.long().to(self.gpu_id))
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = (data.to(self.gpu_id), label.long().to(self.gpu_id))
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def best_model_predictions(self):
        self.model.eval()
        rows = []

        for data, names, labels in self.test_loader:
            data = data.to(self.gpu_id)

            with torch.no_grad():
                output = self.model(data)

            probs = F.softmax(output, dim=1)
            top_probs, top_indices = probs.topk(3, dim=1)  # [batch_size, 3]

            for name, indices in zip(names, top_indices):
                label_str = ' '.join(map(str, indices.tolist()))
                rows.append([name, label_str])

        # Write to CSV
        with open('timesformer_pred_robot_obs.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'labels'])
            writer.writerows(rows)

        print('Saved predictions!')


    def save(self, file_path="./checkpoint"):
        torch.save(self.model.state_dict(), file_path + '.pth')
        #torch.save(self.optimizer.state_dict(), file_path + '_optimizer.pth')

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=self.gpu_id))
        #self.optimizer.load_state_dict(torch.load(file_path + '_optimizer.pth'))



class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


