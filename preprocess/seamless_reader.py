import torch
import torch.nn.functional as F

from torch import Tensor, nn
from fairseq2.data import Collater
import torch.nn.functional as F
from fairseq2.models.sequence import SequenceBatch
from seamless_communication.models.unit_extractor.wav2vec2_layer_output import (
    Wav2Vec2LayerOutputModel,
)
from fairseq2.models.wav2vec2 import Wav2Vec2Model, load_wav2vec2_model
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.unit_extractor.kmeans import KmeansModel
from encodec.utils import convert_audio


class Wav2vecFeatureReader(torch.nn.Module):
    def __init__(
        self, checkpoint_path, kmeans_path, layer=None, dtype = torch.float32, max_chunk=100 * 16_000, lazy_load=False
    ):
        super().__init__()
        # NB: fairseq doesn't support pathlib.Path
        self.model_name_or_card = str(checkpoint_path)
        self.kmeans_url = str(kmeans_path)
        self.should_normalize = False
        self.lazy_load = lazy_load
        self.model = None
        self.out_layer_number = layer - 1
        self.max_chunk = max_chunk
        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=dtype).cuda())
        if not self.lazy_load:
            self.load_checkpoint_()

    @torch.no_grad()  # otherwise some non-leaf nodes appear which breaks serialization
    def load_checkpoint_(self):
        wav2vec2_model = load_wav2vec2_model(
            self.model_name_or_card, device=self.device, dtype=self._float_tensor.dtype
        )
        wav2vec2_model.eval()
        assert isinstance(wav2vec2_model, Wav2Vec2Model)
        self.model = Wav2Vec2LayerOutputModel(wav2vec2_model)
        self.kmeans_model = KmeansModel(self.kmeans_url, self.device, self._float_tensor.dtype)
        self.collate = Collater(pad_value=1, pad_to_multiple=2)
        
    @property
    def device(self):
        return self._float_tensor.device

    @property
    def expected_sample_rate(self) -> int:
        return 16_000

    def forward(self, x, sr):
        if self.lazy_load and self.model is None:
            self.load_checkpoint_()

        return self.get_features(x, sr)

    @torch.inference_mode()
    def get_features(self, inputs, sr):
        inputs = convert_audio(inputs.view(1,-1), sr, 16000, 1)
        inputs = inputs.view(1, -1)
        inputs = F.layer_norm(inputs, inputs.shape)
        
        inputs = inputs.type_as(self._float_tensor)
        
        if inputs.size(1) > self.max_chunk:
            print("too long:", inputs.size(1) / 16000, "s")
        
        feat = []
        for start in range(0, inputs.size(1), self.max_chunk):
            x_chunk = inputs[:, start : start + self.max_chunk]
            if x_chunk.shape[1] < 400: # too short would raise kernal error
                continue
            
            decoded_audio = {
                "waveform": x_chunk.squeeze(0),
                "sample_rate": 16000,
                "format": -1,
            }
            src = self.collate(decoded_audio)["waveform"]
            seqs, padding_mask = get_seqs_and_padding_mask(src)
            seqs = seqs.view(1, -1).to(self.device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask)
            features = self.model(batch, self.out_layer_number).squeeze(0)
            units = self.kmeans_model(features)
            feat.append(units.unsqueeze(0).cpu())
        
        #units, durations = torch.unique_consecutive(units, return_counts=True)
        
        item = {
            "units": torch.cat(feat, dim = 1).squeeze(0),  #no reuduce
        }
        return item
    
