from encodec import EncodecModel
from encodec.utils import convert_audio
import torch

class EncodecFeatureReader(torch.nn.Module):
    def __init__(
        self, bandwidth=6.0, max_chunk=100 * 24_000, repository=None, lazy_load=False, 
    ):
        super().__init__()
        # NB: fairseq doesn't support pathlib.Path
        self.lazy_load = lazy_load
        self.model = None
        self.bandwidth = bandwidth
        self.max_chunk = max_chunk
        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float).cuda())
        if not self.lazy_load:
            self.load_checkpoint_(repository)
            
    @torch.no_grad()  # otherwise some non-leaf nodes appear which breaks serialization
    def load_checkpoint_(self, repository=None):
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_24khz(repository=repository)
        # The number of codebooks used will be determined bythe bandwidth selected.
        # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
        # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
        # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
        # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
        model.set_target_bandwidth(self.bandwidth)
        self.model = model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
            
    @property
    def device(self):
        return self._float_tensor.device
    
    @property
    def expected_sample_rate(self) -> int:
        return 24_000
    
    def forward(self, x, sr):
        if self.lazy_load and self.model is None:
            self.load_checkpoint_()

        return self.get_features(x, sr)
    
    @torch.inference_mode()
    def get_features(self, x, sr):
        # Load and pre-process the audio waveform
        x = convert_audio(x.view(1, -1), sr, self.model.sample_rate, self.model.channels)
        x = x.unsqueeze(0).to(self.device)
        
        if x.size(-1) > self.max_chunk:
            print("too long:", x.size(-1) / self.model.sample_rate, "s")

        # Extract discrete codes from EnCodec
        feat = []
        for start in range(0, x.size(-1), self.max_chunk):
            encoded_frames = self.model.encode(x[...,start : start + self.max_chunk])
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            feat.append(codes.squeeze(0).cpu())
        
        return {"codec": torch.cat(feat, dim=1)}
