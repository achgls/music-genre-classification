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
kaggle datasets download -d achgls/gtzan-music-genre
```

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
---

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

<a id="3">[3]</a> 
torchvision: an audio library for PyTorch, 2021.