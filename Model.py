import torch.nn as nn
from numpy import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from functions import quantize_gabor_number, get_modified_kernel, get_modified_kernel_with_precision

class GaborFilterBank(nn.Module):
    def __init__(self, 
                 num_filters=16, 
                 kernel_size=(5, 5), 
                 orientations=4, 
                 lambd=6.2, 
                 sigma=5.8, 
                 gamma_init=0.5, 
                 psi_init=0.0,
                 quantize_bit_widths=False,
                 clusters=False,
                 theta=None):
        super(GaborFilterBank, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.orientations = orientations
        self.psi_init = psi_init
        self.quantize_bit_widths = quantize_bit_widths
        self.clusters = clusters

        if self.num_filters == 4:
            self.lambd = 6
            self.sigma = 5
            self.gamma_init = 0.55
            self.num_clusters = 3

        elif self.num_filters == 8:
            self.lambd = 6.8
            self.sigma = 5.8
            self.gamma_init = 0.65
            self.num_clusters = 2

            self.bit_widths = [2,3,2,4,2]
            # self.bit_widths = [3,3,3,3]

        elif self.num_filters == 12:
            self.lambd = 6.2
            self.sigma = 5.2
            self.gamma_init = 0.45
            self.num_clusters = 3
        
        # elif self.num_filters == 13:
        #     self.lambd = 6.2
        #     self.sigma = 5.2
        #     self.gamma_init = 0.45

        elif self.num_filters == 16:
            self.lambd = 6.2
            self.sigma = 5.8
            self.gamma_init = 0.50
            self.num_clusters = 2

            self.bit_widths = [2,2]
        
        if theta is None:
            # Default angles (in radians)
            # self.theta = np.array([np.deg2rad(angle) for angle in [45, 80, 135, 180]], dtype=np.float32)
            self.theta = np.deg2rad(np.arange(0, 180, 180/num_filters), dtype=np.float32)
        else:
            if isinstance(theta, (list, tuple, np.ndarray)):
                assert len(theta) == num_filters, f"Theta list must be of length {num_filters}"
                self.theta = np.array(theta, dtype=np.float32)
            else:
                raise ValueError("theta must be a list, tuple, or ndarray of length equal to num_filters")

        # Create Gabor filter kernels using OpenCV
        self.filters = self._create_gabor_kernels()

    def _create_gabor_kernels(self):
        """Creates a list of Gabor kernels using OpenCV."""
        filters = []
        for i in range(self.num_filters):
            filters.append(self._get_gabor_kernel(self.sigma, self.theta[i], self.lambd, self.gamma_init))
        return nn.Parameter(torch.stack(filters), requires_grad=False)  # No learning on the Gabor filters

    def _get_gabor_kernel(self, sigma, theta, lambd, gamma):
        """Generate a single Gabor kernel using OpenCV."""
        ksize = self.kernel_size
        psi = self.psi_init

        # OpenCV function to create the Gabor kernel
        kernel = cv2.getGaborKernel(
            ksize=ksize,
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=psi,
            ktype=cv2.CV_32F
        )

        # to use for different bit widths and clustering

        # 4
        # bit_widths = [3,5,3,5,3]
        # bit_widths = [2,3,3,5,3]
        # bit_widths = [4,3]

        # 8
        # bit_widths = [2,3,2,4,2]
        # bit_widths = [5,4,3,2,6]
        # bit_widths = [3,3,3,3]

        # 12
        # bit_widths = [3,6,2,2,3]
        # bit_widths = [2,3,2,4,6]
        # bit_widths = [3,5,2,2]

        # 16
        # bit_widths = [2,4,2,4,2]
        # bit_widths = [2,4,5,3,3]
        # bit_widths = [2,2]

        # #13
        # bit_widths = [
        #     [2,4,4,2,3],
        #     [4,2,5,5,3],
        #     [4,2,2,2,4],
        #     [3,5,5,2,4],
        #     [3,2,4,4,2]
        # ]

        #4
        # bit_widths = [
        #     [3,3,3,6,3],
        #     [4,3,2,5,5],
        #     [2,2,2,2,2],
        #     [5,5,2,3,4],
        #     [3,6,3,3,3]
        # ]

        if self.clusters and self.quantize_bit_widths:
            kernel = get_modified_kernel_with_precision(kernel,len(self.bit_widths),ksize,self.bit_widths)

        elif self.quantize_bit_widths:
            print(self.bit_widths)
            modified_kernel = np.zeros(ksize)
            for i in range(ksize[0]):
                for j in range(ksize[1]):
                    modified_kernel[i][j] = quantize_gabor_number(kernel[i][j],self.bit_widths[i])
            kernel = modified_kernel
        
        elif self.clusters:
            kernel = get_modified_kernel(kernel, self.num_clusters, ksize)

        kernel1 = kernel.astype(np.float16)
        print()

        # Convert the kernel to a torch tensor and ensure correct dimensions
        kernel = torch.from_numpy(kernel1).float()  # Convert to float tensor
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions (1, 1, h, w)
        return kernel

    def forward(self, x):
        """Applies Gabor filter bank to input images."""
        filters = self.filters.to(x.device)  # Ensure filters are on the same device as the input
        batch_size = x.size(0)
        
        # Apply each Gabor filter to the input using 2D convolution
        filtered_outputs = []
        for i in range(self.num_filters):
            filtered_outputs.append(F.conv2d(x, filters[i], stride=1, padding=2))  # Padding is based on kernel size

        return torch.cat(filtered_outputs, dim=1)  # Concatenate filters along the channel axis

    def get_config(self):
        """Return configuration for serialization."""
        return {
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'orientations': self.orientations,
            'lambd': self.lambd,
            'sigma': self.sigma,
            'gamma_init': self.gamma_init,
            'psi_init': self.psi_init,
            'theta': self.theta.tolist()
        }

    @classmethod
    def from_config(cls, config):
        """Create a layer from its configuration."""
        config['theta'] = np.array(config['theta'], dtype=np.float32)
        return cls(**config)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride, padding, ratio, reduction=4,
    ):
        super(MBBlock, self).__init__()
        # self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * ratio)
        self.expand = in_channels != hidden_dim

        # This is for squeeze and excitation block
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                kernel_size=3,stride=1,padding=1)

        self.conv = nn.Sequential(
                ConvBlock(hidden_dim,hidden_dim,kernel_size,
                  stride,padding,groups=hidden_dim),
                SqueezeExcitation(hidden_dim, reduced_dim),
                nn.Conv2d(hidden_dim, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        if self.expand:
          x = self.expand_conv(inputs)
        else:
          x = inputs
        return self.conv(x)



class EfficientNet(nn.Module):
    def __init__(self,num_filters,quantize,cluster,output=1280):
        super(EfficientNet, self).__init__()
        phi, resolution, dropout = (0, 224, 0.2)
        self.depth_factor, self.width_factor = 1.2**phi, 1.1**phi
        self.last_channels = int(ceil(1280 * self.width_factor))
        self.avgpool= nn.AdaptiveAvgPool2d(1)
        self.feature_extractor()
        self.flatten = nn.Flatten()

        self.input_conv = nn.Conv2d(num_filters, 3, kernel_size=1, padding=1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channels, output),
            # nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(output),
            nn.Linear(1280,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )
        
        self.gabor_filter_bank = GaborFilterBank(
            num_filters=num_filters,
            kernel_size=(5,5),
            orientations=4,
            quantize_bit_widths=quantize,
            clusters=cluster
        )

    def feature_extractor(self):
        channels = int(32 * self.width_factor)
        features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels
        basic_mb_params = [
            # k, channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        for k, c_o, repeat, s, n in basic_mb_params:
            # For numeric stability, we multiply and divide by 4
            out_channels = int(4 * ceil(int(c_o * self.width_factor) / 4))
            num_layers = ceil(repeat * self.depth_factor)
            num_layers = int(num_layers)

            for layer in range(num_layers):
                if layer == 0:
                  stride = s
                else:
                  stride = 1
                features.append(
                        MBBlock(in_channels,out_channels,ratio=k,
                        stride=stride,kernel_size=n,padding=n// 2)
                    )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, self.last_channels, 
            kernel_size=1, stride=1, padding=0)
        )
        self.extractor = nn.Sequential(*features)

    def forward(self, x):
        x = self.gabor_filter_bank(x)
        x = self.input_conv(x)
        x = self.avgpool(self.extractor(x))
        return self.classifier(self.flatten(x))
    