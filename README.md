# Hero Name Recognition

<div align="center">
    <img src="img/logo_image.jpg" width="100%"/>
</div>

This repository provides a CNN baseline (backbone from [timm](https://github.com/huggingface/pytorch-image-models)) for Hero Name Recognition problem in a game called League of Legends: Wild Rift (a.k.a Wild Rift).  
This game is an mutiplayer online battle arena game developed by Riot Games. In the highlight moment detection systems, it is important to recognize the hero appearing on the message bar when a battle happens.

For example:
| Image | Hero |
|--|--|
| ![](img/example1.jpg) | Ahri |
| ![](img/example2.jpg) | Ashe |


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Dependencies

```
pip3 install -r requirements.txt
```

### Data

Put your data into the `data` folder. The structure of the folder will be:
```
./
    data/
        train/
            img1.jpg
            img2.jpg
            ...
        test/
            img1.jpg
            img2.jpg
            ...
```

### Logging

This repository uses [Comet ML](https://www.comet.com/) for experiment tracking. See this for how to get started and log in with your account. Then, go to train.py and modify line 17, in the argument `api_key` for your own project.

### Training

You can choose any model from timm to make the backbone. Here is `resnet18`:
```
python3 train.py --backbone resnet18 --e 100 --b 64 
```

### Prediction

Make sure the `--backbone` to predict is the same as the training.

```
python3 predict.py --backbone resnet18 --d <path/to/test/data> --c <path/to/weight>
```

For quickly prediction without training, download the trained resnet18 weight [here](https://drive.google.com/file/d/1rttwrLbF3fFOT5HvorXDslUPeDgoniOF/view?usp=sharing) into `checkpoint` folder and run:
```
python3 predict.py --backbone resnet18 --d data/train --c checkpoint/weight_42_3.7745168209075928.pt
```

> Use the flag -h to see other training or inference arguments.

## Author
Nguyen Huy An  
Email: anhuynguyen001@gmail.com