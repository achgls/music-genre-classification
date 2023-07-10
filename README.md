Music genre classification with SENets
========================================
Music genre classification projet as part of the Numerical Analysis for Machine Learning course at Politecnico di Milano, A.Y. 2022-2023.
 

Dataset
-------
The data used for this project can be found
[here](https://www.kaggle.com/datasets/achgls/gtzan-music-genre) 
as a Kaggle dataset. It can be downloaded into the `res` directory
directly from the terminal using
[Kaggle's CLI tool](https://www.kaggle.com/docs/api):
```console
kaggle datasets download achgls/gtzan-music-genre -p ./res --unzip
```
The GTZAN dataset was originally introduced in the following paper by George Tzanetakis (hence
the name) in 2002 as part of his Ph.D. thesis work.
```bibtex
@ARTICLE{1021072,
  author={Tzanetakis, G. and Cook, P.},
  journal={IEEE Transactions on Speech and Audio Processing}, 
  title={Musical genre classification of audio signals}, 
  year={2002},
  volume={10},
  number={5},
  pages={293-302},
  doi={10.1109/TSA.2002.800560}}
```


Reproducibility
---------------
In order to allow for proper uncontaminated assessment of each parameter's impact
on training, the training script present in this repo allows for full reproducibility of experiments.
When modifying a certain parameter to evaluate its impact on training, you can thus be certain
that all other parameters remain stable.
When a seed is given as argument to the script, the model is prevented to use any
non-deterministic operations, and the seed is set as the pseudo-random
number generator for the initialization of models weights as well as data sampling.
You might get a `RuntimeError` from NVIDIA backend when trying to run
reproducible experiments, in that case you will have to set an environment variable as so:
```console
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Requirements
------------
Libraries used in this project are listed in [requirements.txt](requirements.txt) and
can be installed at once with:
```console
pip install -r requirements.txt
```
In addition to those,
you need a torchaudio-compatible audio backend installed. This would be `soundfile`
for Windows machines: `pip install soundfile`, and `sox_io` for Unix systems:
`pip install sox`. More info on bakends are available
on the [PyTorch audio backends documentation](https://pytorch.org/audio/stable/backend.html).

References
-----------
Paper suggested as a guideline for the project:

<a id="1">[1]</a> 
Xu, Yijie and Zhou, Wuneng, 2020.
[A deep music genres classification model based on CNN with Squeeze & Excitation Block](https://ieeexplore.ieee.org/document/9306374).
Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 332-338.
```bibtex
@INPROCEEDINGS{9306374,
  author={Xu, Yijie and Zhou, Wuneng},
  booktitle={2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
  title={A deep music genres classification model based on CNN with Squeeze & Excitation Block}, 
  year={2020}
}
```
----------

<a id="2">[2]</a> 
torchaudio: an audio library for PyTorch, 2021.
```bibtex
@article{yang2021torchaudio,
  title={TorchAudio: Building Blocks for Audio and Speech Processing},
  author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent Quenneville-Bélair and Yangyang Shi},
  journal={arXiv preprint arXiv:2110.15018},
  year={2021}
}
```

[//]: # (<a id="3">[3]</a> )

[//]: # (torchvision: computer vision in PyTorch, 2016.)

[//]: # (```bibtex)

[//]: # (@software{TorchVision_maintainers_and_contributors_TorchVision_PyTorch_s_Computer_2016,)

[//]: # (    author = {TorchVision maintainers and contributors},)

[//]: # (    license = {BSD-3-Clause},)

[//]: # (    month = nov,)

[//]: # (    title = {{TorchVision: PyTorch's Computer Vision library}},)

[//]: # (    url = {https://github.com/pytorch/vision},)

[//]: # (    year = {2016})

[//]: # (})

[//]: # (```)