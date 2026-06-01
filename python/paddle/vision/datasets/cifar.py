#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import tarfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image

import paddle
from paddle.dataset.common import _check_exists_and_download
from paddle.io import Dataset

if TYPE_CHECKING:
    import numpy.typing as npt

    from paddle._typing.dtype_like import _DTypeLiteral
    from paddle.vision.transforms.transforms import _Transform

    from ..image import _ImageBackend, _ImageDataType

    _DatasetMode = Literal["train", "test"]

__all__ = []

URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-binary.tar.gz'
CIFAR10_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'
CIFAR100_URL = URL_PREFIX + 'cifar-100-binary.tar.gz'
CIFAR100_MD5 = '03b5dce01913d631647c71ecec9e9cb8'

MODE_FLAG_MAP = {
    'train10': (
        'data_batch_1.bin',
        'data_batch_2.bin',
        'data_batch_3.bin',
        'data_batch_4.bin',
        'data_batch_5.bin',
    ),
    'test10': ('test_batch.bin',),
    'train100': ('train.bin',),
    'test100': ('test.bin',),
}

CIFAR10_RECORD_SIZE = 3073
CIFAR100_RECORD_SIZE = 3074
IMAGE_SIZE = 3072


class Cifar10(Dataset[tuple["_ImageDataType", "npt.NDArray[Any]"]]):
    """
    Implementation of `Cifar-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 10 categories.

    Args:
        data_file (str|None, optional): Path to data file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/cifar
        mode (str, optional): Either train or test mode. Default 'train'.
        transform (Callable|None, optional): transform to perform on image, None for no transform. Default: None.
        download (bool, optional): download dataset automatically if :attr:`data_file` is None. Default True.
        backend (str|None, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}.
            If this option is not set, will get backend from :ref:`paddle.vision.get_image_backend <api_paddle_vision_get_image_backend>`,
            default backend is 'pil'. Default: None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of Cifar10 dataset.

    Examples:

        .. code-block:: pycon

            >>> # doctest: +TIMEOUT(60)
            >>> import itertools
            >>> import paddle.vision.transforms as T
            >>> from paddle.vision.datasets import Cifar10

            >>> cifar10 = Cifar10()
            >>> print(len(cifar10))
            50000

            >>> for i in range(5):  # only show first 5 images
            ...     img, label = cifar10[i]
            ...     # do something with img and label
            ...     print(type(img), img.size, label)
            ...     # <class 'PIL.Image.Image'> (32, 32) 6


            >>> transform = T.Compose(
            ...     [
            ...         T.Resize(64),
            ...         T.ToTensor(),
            ...         T.Normalize(
            ...             mean=[0.5, 0.5, 0.5],
            ...             std=[0.5, 0.5, 0.5],
            ...             to_rgb=True,
            ...         ),
            ...     ]
            ... )
            >>> cifar10_test = Cifar10(
            ...     mode="test",
            ...     transform=transform,  # apply transform to every image
            ...     backend="cv2",  # use OpenCV as image transform backend
            ... )
            >>> print(len(cifar10_test))
            10000

            >>> for img, label in itertools.islice(iter(cifar10_test), 5):  # only show first 5 images
            ...     # do something with img and label
            ...     print(type(img), img.shape, label)  # type: ignore
            ...     # <class 'paddle.Tensor'> [3, 64, 64] 3

    """

    mode: _DatasetMode
    backend: _ImageBackend
    data_file: str | None
    transform: _Transform[Any, Any] | None
    dtype: _DTypeLiteral

    def __init__(
        self,
        data_file: str | None = None,
        mode: _DatasetMode = 'train',
        transform: _Transform[Any, Any] | None = None,
        download: bool = True,
        backend: _ImageBackend | None = None,
    ) -> None:
        assert mode.lower() in [
            'train',
            'test',
        ], f"mode.lower() should be 'train' or 'test', but got {mode}"
        self.mode = mode.lower()

        if backend is None:
            backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                f"Expected backend are one of ['pil', 'cv2'], but got {backend}"
            )
        self.backend = backend

        self._init_url_md5_flag()

        self.data_file = data_file
        if self.data_file is None:
            assert download, (
                "data_file is not set and downloading automatically is disabled"
            )
            self.data_file = _check_exists_and_download(
                data_file, self.data_url, self.data_md5, 'cifar', download
            )

        self.transform = transform

        # read dataset into memory
        self._load_data()

        self.dtype = paddle.get_default_dtype()

    def _init_url_md5_flag(self):
        self.data_url = CIFAR10_URL
        self.data_md5 = CIFAR10_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '10']
        self.record_size = CIFAR10_RECORD_SIZE

    def _get_label(self, record):
        return int(record[0])

    def _load_data(self):
        self.data = []
        with tarfile.open(self.data_file, mode='r') as f:
            members = {member.name: member for member in f.getmembers()}
            for expected_name in self.flag:
                member = next(
                    (
                        members[name]
                        for name in sorted(members)
                        if name.endswith('/' + expected_name)
                        or name == expected_name
                    ),
                    None,
                )
                if member is None:
                    raise ValueError(
                        f"Expected CIFAR binary archive member {expected_name} "
                        f"was not found in {self.data_file}"
                    )
                fileobj = f.extractfile(member)
                assert fileobj is not None
                data = fileobj.read()
                if len(data) % self.record_size != 0:
                    raise ValueError(
                        f"CIFAR binary file {member.name} size {len(data)} is "
                        f"not divisible by record size {self.record_size}"
                    )
                records = np.frombuffer(data, dtype=np.uint8).reshape(
                    -1, self.record_size
                )
                for record in records:
                    self.data.append(
                        (record[-IMAGE_SIZE:].copy(), self._get_label(record))
                    )

    def __getitem__(self, idx: int) -> tuple[_ImageDataType, npt.NDArray[Any]]:
        image, label = self.data[idx]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, np.array(label).astype('int64')

        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)


class Cifar100(Cifar10):
    """
    Implementation of `Cifar-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 100 categories.

    Args:
        data_file (str|None, optional): path to data file, can be set None if
            :attr:`download` is True. Default: None, default data path: ~/.cache/paddle/dataset/cifar
        mode (str, optional): Either train or test mode. Default 'train'.
        transform (Callable|None, optional): transform to perform on image, None for no transform. Default: None.
        download (bool, optional): download dataset automatically if :attr:`data_file` is None. Default True.
        backend (str|None, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}.
            If this option is not set, will get backend from :ref:`paddle.vision.get_image_backend <api_paddle_vision_get_image_backend>`,
            default backend is 'pil'. Default: None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of Cifar100 dataset.

    Examples:

        .. code-block:: pycon

            >>> # doctest: +TIMEOUT(60)
            >>> import itertools
            >>> import paddle.vision.transforms as T
            >>> from paddle.vision.datasets import Cifar100

            >>> cifar100 = Cifar100()
            >>> print(len(cifar100))
            50000

            >>> for i in range(5):  # only show first 5 images
            ...     img, label = cifar100[i]
            ...     # do something with img and label
            ...     print(type(img), img.size, label)
            ...     # <class 'PIL.Image.Image'> (32, 32) 19


            >>> transform = T.Compose(
            ...     [
            ...         T.Resize(64),
            ...         T.ToTensor(),
            ...         T.Normalize(
            ...             mean=[0.5, 0.5, 0.5],
            ...             std=[0.5, 0.5, 0.5],
            ...             to_rgb=True,
            ...         ),
            ...     ]
            ... )

            >>> cifar100_test = Cifar100(
            ...     mode="test",
            ...     transform=transform,  # apply transform to every image
            ...     backend="cv2",  # use OpenCV as image transform backend
            ... )
            >>> print(len(cifar100_test))
            10000

            >>> for img, label in itertools.islice(iter(cifar100_test), 5):  # only show first 5 images
            ...     # do something with img and label
            ...     print(type(img), img.shape, label)  # type: ignore
            ...     # <class 'paddle.Tensor'> [3, 64, 64] 49

    """

    def __init__(
        self,
        data_file: str | None = None,
        mode: _DatasetMode = 'train',
        transform: _Transform[Any, Any] | None = None,
        download: bool = True,
        backend: _ImageBackend | None = None,
    ) -> None:
        super().__init__(data_file, mode, transform, download, backend)

    def _init_url_md5_flag(self):
        self.data_url = CIFAR100_URL
        self.data_md5 = CIFAR100_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '100']
        self.record_size = CIFAR100_RECORD_SIZE

    def _get_label(self, record):
        return int(record[1])
