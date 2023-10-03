import torch
import torch.nn as nn

# YOLO Convolutional layers architecture from paper
# Tuples shape: (kernal_size, num_filters, stride, padding)
architecture_config = [
        (7, 64, 2, 3),
        "M", # Max pool layer
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4], # Iterating Tuples 4 times 
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) # Not included in original paper, since it wasn't invented at the time. But there's no point for not including it now.
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    '''
    Yolo version 1 model with slight adjustments compared to original paper.

    Note: Expects image input shape: (2, 3, 448, 448).
    '''
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture) # darknet comes from paper and was implemented by Joseph Redmon (author of paper) 
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        '''
        Creates Convolutional layers from architecture_config.

        Parameters:
            architecture (dict): containing model architecture in own defined format.

        Returns: 
            nn.Sequential with the Convolutional network architecture. 
        '''
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if isinstance(x, tuple):
                # Add Convolutional Block with configuration from tuple 
                layers += [
                        CNNBlock(
                            in_channels = in_channels,
                            out_channels = x[1],
                            kernel_size = x[0],
                            stride = x[2],
                            padding = x[3],
                        )
                    ]
                in_channels = x[1] 
            elif isinstance(x, list):
                # Add iterating Conv block with configuration from list with tuples 
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repetitions = x[2] # Integer
                for _ in range(num_repetitions):
                    layers += [
                            CNNBlock(
                                in_channels = in_channels,
                                out_channels = conv1[1],
                                kernel_size = conv1[0],
                                stride = conv1[2],
                                padding = conv1[3],
                            )
                        ]
                    layers += [
                            CNNBlock(
                                in_channels = conv1[1], # out_channels of conv1
                                out_channels = conv2[1],
                                kernel_size = conv2[0],
                                stride = conv2[2],
                                padding = conv2[3],
                            )
                        ]
                    in_channels = conv2[1] # out_channel of second CNNBlock, so the in_channels is correct for next iteration
            elif isinstance(x, str) and x == "M":
                # Add max pooling layer
                layers += [
                        nn.MaxPool2d(
                            kernel_size = (2,2),
                            stride = (2,2),
                        )
                    ]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        '''
        Creates the fully connected layers.

        Parameters:
            split_size (integer): into how many cells the image should be split. Number of cells = S * S.
            num_boxes (integer): how many bboxes are predicted for each image cell. 
            num_classes (integer): how many classes there are to predict.

        Returns:
            nn.Sequential of the fully connected layers.
            Note: output has to be reshaped to (split_size, split_size, 30) externally!
        '''
        return nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024 * split_size * split_size, 496), # In original paper 4960 is used.
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5)), # will be reshaped to (S, S, 30) where C + B*5 = 30
            )


def test(split_size=7, num_boxes=2, num_classes=20):
    model = Yolov1(split_size = split_size, num_boxes = num_boxes, num_classes = num_classes)
    sample_image_shape = (2, 3, 448, 448)
    x = torch.randn(sample_image_shape)

    expected_prediction_shape = num_classes + num_boxes * 5
    expected_model_output_shape = torch.Size([num_boxes, (split_size * split_size * expected_prediction_shape)])

    output_shape = model(x).shape
    if output_shape != torch.Size([2, 1470]):
        print("Test Failed.")
        print("Output shape:", output_shape)
        print("Expected output shape:", expected_model_output_shape)
        print()
    else:
        print("Test Passed.")
        print("Input:")
        print("\tsplit_size =", split_size)
        print("\tnum_boxes =", num_boxes)
        print("\tnum_classes =", num_classes)
        print("Output shape:", output_shape)
        print()

# test() # Use for sanity checking model works


