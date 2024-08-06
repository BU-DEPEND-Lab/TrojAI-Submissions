from typing import Dict
from pathlib import Path

import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel

class FreqPurification(TrojAIMitigation):
    def __init__(self, device, loss_cls, optim_cls, lr, epochs, batch_size=32, num_workers=1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        self._optimizer_class = optim_cls
        self._loss_cls = loss_cls
        self.lr = lr
        self.epochs = epochs
        self.top_percent = 0.007

        # self.ckpt_dir = ckpt_dir
        # self.ckpt_every = ckpt_every
        # self.gaussian_blur = transforms.GaussianBlur(kernel_size=(19, 19), sigma=(10, 20))
        # self.flip_image = transforms.RandomHorizontalFlip(p=1)  # Always flip the image
        # self.gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 10))
    def filter_high_frequency(self, image, top_percent):
        # Apply DFT to each channel
        dft_image = torch.fft.fft2(image)
        # Compute the magnitude of each frequency component
        magnitude_spectrum = torch.abs(dft_image)
        # Flatten the magnitude spectrum to get a 1D array of magnitudes
        flattened_magnitude = magnitude_spectrum.flatten()
        sorted_magnitude = torch.sort(flattened_magnitude, descending=True).values
        threshold_index = int(len(sorted_magnitude) * top_percent / 100)
        threshold_value = sorted_magnitude[threshold_index]
        # Filter out the high-frequency components
        mask = magnitude_spectrum <= threshold_value
        filtered_dft_image = dft_image * mask
        # inverse DFT 
        filtered_image = torch.fft.ifft2(filtered_dft_image)
        # Take the real part of the inverse DFT result
        filtered_image = filtered_image.real
        return filtered_image

    def preprocess_transform(self, x):
        original_batch_size = x.shape[0]
        x = self.filter_high_frequency(x, self.top_percent)
        # x = self.flip_image(x)
        # processed_x = x

        return x, {"original_batch_size": original_batch_size}


    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Args:
            model: the model to repair
            dataset: a dataset of examples
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        pass
        # model.train()
        # optim = self._optimizer_class(model.parameters(), lr=self.lr)
        # loss_fn = self._loss_cls()
        # trainloader = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     drop_last=True,
        #     pin_memory=True
        # )
        
        # for i in range(self.epochs):
        #     pbar = tqdm(trainloader)
        
        #     for x, y in pbar:
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         optim.zero_grad()
        #         pred = model(x)
        #         loss = loss_fn(pred, y)
        #         loss.backward()
        #         optim.step()
        #         pbar.set_description(f"Epoch: {i} | Loss: {loss}")
            
        # if self.ckpt_every != 0 and i % self.ckpt_every == 0:
        #     ckpt_path = Path(self.ckpt_dir)
        #     ckpt_path.mkdir(exist_ok=True)
        #     torch.save({
        #         'epoch': i,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optim.state_dict(),
        #     }, ckpt_path / Path(f"ft_ckpt_epoch{i + 1}.ckpt"))
        #     print(f"Saved ckpt to {ckpt_path / Path(f'ft_ckpt_epoch{i + 1}.ckpt')}")
            
            

        return TrojAIMitigatedModel(model)
