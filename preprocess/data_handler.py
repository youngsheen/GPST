import pathlib
import logging
import torch
import torchaudio
from typing import List, Iterator
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, chunk_size = 30):
        with open(manifest, "r") as fin:
            self.files = [x.strip() for x in fin.readlines()]
        self.chunk_size = chunk_size
        logger.info(
            f"containing {len(self.files)} files"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio_path = pathlib.Path(self.files[index])
        waveform, sr = torchaudio.load(audio_path.as_posix())
        waveform = waveform.squeeze(0)
        
        waveform_split = waveform.split(self.chunk_size * sr)
        
        if len(waveform_split) != 1:
            waveform_not_chunk = None
            if len(waveform_split[-1]) < 1 * sr: # 1 second 
                waveform_chunk = torch.stack(waveform_split[:-1], dim = 0)
            elif len(waveform_split[-1]) != self.chunk_size * sr :
                waveform_chunk = torch.stack(waveform_split[:-1] + (waveform[-self.chunk_size * sr:],), dim = 0)
            else:
                waveform_chunk = torch.stack(waveform_split, dim = 0)
        else:
            waveform_chunk = None
            waveform_not_chunk = waveform_split[-1]
            if len(waveform_not_chunk) < 1 * sr: # 1 second 
                logger.info(f"{len(waveform_not_chunk)} too short pass")
                waveform_not_chunk = None
            else:
                waveform_not_chunk = waveform_not_chunk.unsqueeze(0)
        
        return {
            "waveform": waveform_chunk, # for batch accelerate
            "waveform_not_chunk": waveform_not_chunk,
            "sample_rate": sr,
            "audio_path": audio_path
            }


class RankBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, rank, world_size, drop_last=False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

    def __len__(self) -> int:
        return len(self.dataset) // self.world_size + int(self.rank < (len(self.dataset) % self.world_size))

    def __iter__(self) -> Iterator[List[int]]:    
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in range(self.rank, len(self.dataset), self.world_size):
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]
            
            
            
def custom_collate_fn(batch):
    new_batch = {}
    
    waveforms = []
    audio_paths = []
    waveform_not_chunks = []
    audio_paths_not_chunks = []
    sample_rates = []
    sample_rates_not_chunks = []
    for smaple in batch:
        if smaple['waveform'] is not None:
            waveforms.append(smaple['waveform'])
            audio_paths.extend([smaple['audio_path'].as_posix() + str(i) for i in range(1, smaple['waveform'].shape[0] + 1)])
            sample_rates.extend([smaple['sample_rate']] * smaple['waveform'].shape[0])
        
        if smaple['waveform_not_chunk'] is not None:
            waveform_not_chunks.append(smaple['waveform_not_chunk'])
            audio_paths_not_chunks.append(smaple['audio_path'].as_posix())
            sample_rates_not_chunks.append(smaple['sample_rate'])

    new_batch['sample_rate'] = sample_rates if len(sample_rates) != 0 else None
    new_batch['waveform'] = torch.cat(waveforms, dim = 0) if len(waveforms) != 0 else None
    new_batch['audio_path'] = audio_paths if len(audio_paths) != 0 else None
    new_batch['sample_rate_not_chunk'] = sample_rates_not_chunks if len(sample_rates_not_chunks) != 0 else None
    new_batch['waveform_not_chunk'] = waveform_not_chunks if len(waveform_not_chunks) != 0 else None
    new_batch['audio_paths_not_chunk'] = audio_paths_not_chunks if len(audio_paths_not_chunks) != 0 else None
    return new_batch