import numpy as np
import torch
from torch import nn
from torchvision import models
import cv2
from transformers import AutoModel
from transformers import AutoConfig
import torchvision.models as models
from resnet import BasicBlock,ResNet,model_urls,Bottleneck

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import copy
class Multimodel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

        self.resnets = list()
        self.nlpmodel=list()
        config=AutoConfig.from_pretrained("")
        self.bert_model = AutoModel.from_config(config)
        self.resnet_model = ResNet(Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Linear((768+2048)*self.n, 2)

    def forward(self, x):
	pass


