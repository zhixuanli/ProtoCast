from .convlstm import ConvLSTM
from .predrnn import PredRNN
from .mau import MAU
from .simvp import SimVP
from .tau import TAU
from .ITS.simvp import SimVP as Met2Net
from .predformer import PredFormer

method_maps = {
    'convlstm': ConvLSTM, 'predrnn': PredRNN, 'mau': MAU, 'simvp': SimVP,
    'tau': TAU, 'its': Met2Net, 'predformer': PredFormer,
}
