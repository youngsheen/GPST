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
        self, checkpoint_path, kmeans_path, layer=None, dtype = torch.float32, max_chunk=60 * 16_000, lazy_load=False, encoder_max_sample=None
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
        self.encoder_max_sample = encoder_max_sample
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
        inputs = [convert_audio(xx.view(1,-1), ss, self.expected_sample_rate, 1) for xx, ss in zip(inputs, sr)]
        inputs = [F.layer_norm(x, x.shape).type_as(self._float_tensor) for x in inputs]
                        
        output = []
        for idx in range(0, len(inputs), self.encoder_max_sample):
            decoded_audio = [{
                    "waveform": x.squeeze(0),
                    "sample_rate": 16000,
                    "format": -1,
                } for x in inputs[idx:idx + self.encoder_max_sample]]
            src = self.collate(decoded_audio)["waveform"]
            seqs, padding_mask = get_seqs_and_padding_mask(src)
            seqs = seqs.view(len(inputs[idx:idx + self.encoder_max_sample]), -1).to(self.device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask)
            features = self.model(batch, self.out_layer_number)
            features_flatten = features.flatten(0,1)
            units = self.kmeans_model(features_flatten)
            units = units.view(features.shape[0],-1)
            output.append(units.cpu())
        
        #units, durations = torch.unique_consecutive(units, return_counts=True)
        item = {"tokens": torch.cat(output, dim = 0)}
        return item
    
