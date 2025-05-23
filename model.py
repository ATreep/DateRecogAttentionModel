import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import einops
from nmt_utils import *

torch.set_default_device(torch.device("cpu"))  # This model is fast to forward even on cpu

# Dictionaries for converting text to indices
human_vocab, machine_vocab, inv_machine_vocab = load_vocab()

# Some hyper parameters for attention model
Tx = 30  # Assuming this is the max length of human-readable date
Ty = 10  # Because "YYYY-MM-DD" is 10 characters
n_a = 32  # Size of hidden state for Pre-attention Bi-LSTM
n_s = 64  # Size of hidden state for Post-attention LSTM

class WeightedSum(nn.Module):
    """
    Compute context_t by weighted sum
    """
    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, alpha):
        return torch.sum(alpha * self.a, dim=1)

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, s_prev: torch.Tensor, a: torch.Tensor):
        """
        Params:
            s_prev: previous hidden state of LSTM, of shape (1, m, n_s) needed to reshape to (m, n_s)
            a: Bi-LSTM's outputs, of shape (m, Tx, 2*n_a), each output is a concatenation of two direction outputs

        Return:
            context_t, output of attention block, of shape (m, 1, 2 * n_a)
        """
        Tx = a.shape[1]
        s_prev = s_prev.reshape(-1, n_s)
        s_prev_repeated = einops.repeat(s_prev, "m n_s -> m Tx n_s", Tx=Tx)
        concat = torch.cat((s_prev_repeated, a), dim=2)  # Shape: (m, Tx, n_s + 2 * n_a)

        return nn.Sequential(
            nn.Linear(in_features=n_s + 2 * n_a, out_features=10),
            nn.Tanh(),  # Intermedia variable e
            nn.Linear(in_features=10, out_features=1),
            nn.ReLU(),  # variable `energies`
            nn.Softmax(dim=2),  # alphas, of shape (m, Tx, 1)
            WeightedSum(a)  # sum, of shape (m, 2 * n_a)
        )(concat).reshape(-1, 1, 2*n_a)  # context, of shape (m, 1, 2 * n_a)
    
class NMTAttentionModel(nn.Module):
    """
    Output:
        Predicted result sequence without softmax, of shape (m, Ty, len(machine_vocab))

    """
    def __init__(self, Tx: int, Ty: int):
        super().__init__()
        self.pre_attention_lstm = nn.LSTM(1, n_a, 1, batch_first=True, bidirectional=True)  # whose output obeys AttentionBlock's requirement
        self.attention_block = AttentionBlock()
        self.post_attention_lstm = nn.LSTM(2 * n_a, n_s, 1, batch_first=True)
        self.output_layer = nn.Linear(in_features=n_s, out_features=len(machine_vocab))

    def forward(self, X, example_size: int):
        s_t = torch.zeros((1, example_size, n_s))
        c_t = torch.zeros((1, example_size, n_s))
        a, _ = self.pre_attention_lstm(X, (torch.zeros((2, example_size, n_a)), torch.zeros((2, example_size, n_a))))  # shape (m, Tx, 2*n_a)
        y_pred = torch.zeros((example_size, Ty, len(machine_vocab)))

        for t in range(Ty):
            context_t = self.attention_block(s_t, a)  # shape (m, 1, 2 * n_a)
            _, (s_t, c_t) = self.post_attention_lstm(context_t, (s_t, c_t))  # We do not need outputs from post attention lstm
            output = self.output_layer(s_t)  # shape (m, len(machine_vocab))
            y_pred[:, t] = output

        return y_pred
    
model = NMTAttentionModel(Tx, Ty)
model.load_state_dict(torch.load("date_recognition_attention_model_cpu.pt", weights_only=True))  # Pre-trained weights under Adam optimizer with 1 million samples
model.eval()

def test_model(model, human_date):
    source = string_to_int(human_date, Tx, human_vocab)
    source = torch.tensor(source, dtype=torch.float32).reshape(1, -1, 1)
    with torch.no_grad():
        pred = model(source, 1)
    prediction = pred.argmax(dim=2).reshape(Ty,)
    output = ''.join([inv_machine_vocab[int(i)] for i in prediction])
    
    return output