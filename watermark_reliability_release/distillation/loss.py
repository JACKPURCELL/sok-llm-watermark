import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

class custom_loss(nn.Module):
    def __init__(self, temperature):
        super(custom_loss, self).__init__()
        self.temperature = temperature

    def forward(self,
                teacher_logits: Tensor,
                student_logits: Tensor,
                labels: Tensor) -> Tensor:
        """
        The distillation loss for distilating a BERT-like model.
        The loss takes the (teacher_logits), (student_logits) and (labels) for various losses.
        """
        # Temperature and sotfmax
        student_logits, teacher_logits = (student_logits / self.temperature).softmax(1), (
                teacher_logits / self.temperature).softmax(1)
        # Classification loss (problem-specific loss)
        loss = CrossEntropyLoss()(student_logits, labels)
        # CrossEntropy teacher-student loss
        loss = loss + CrossEntropyLoss()(student_logits, teacher_logits)
        # Half loss from hard loss and half from soft loss
        loss = loss / 2
        return loss