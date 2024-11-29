# VectorPainter

## 🔥Quickstart

### Starry

```shell
CUDA_VISIBLE_DEVICES=7 python vectorpainter.py x=stroke prompt='A horse is drinking water by the lake. van Gogh style.' target="./assets/starry.jpg" result_path='./workspace/Starry/horse_starry' seed=100
CUDA_VISIBLE_DEVICES=6 python vectorpainter.py x=stroke prompt='A painting of a cat.' target="./assets/starry.jpg" result_path='./workspace/Starry/cat_starry' seed=8019
```

### Sunflowers

```shell
CUDA_VISIBLE_DEVICES=1 python vectorpainter.py x=stroke "prompt='A bouquet of roses in a vase.'" target="./assets/sunflowers.jpg" result_path='./workspace/sunflowers/roses_sunflowers'
CUDA_VISIBLE_DEVICES=2 python vectorpainter.py x=stroke "prompt='A fire-breathing dragon.'" target="./assets/sunflowers.jpg" result_path='./workspace/sunflowers/dragon_sunflowers' seed=100
CUDA_VISIBLE_DEVICES=1 python vectorpainter.py x=stroke "prompt='A bouquet of roses in a vase.'" target="./assets/sunflowers.jpg" x.pos_type='bez' x.pos_loss_weight=0.5 result_path='./workspace/sunflowers/roses_sunflowers_bez0.5' seed=100
```

### Field

```shell
python vectorpainter.py "prompt='A brightly colored mushroom growing on a log. van Gogh style.'" x=stroke target="./assets/Field.jpg" result_path='./workspace/Field/mushroom_Field' seed=951222
```

### Impressionism

```shell
CUDA_VISIBLE_DEVICES=5 python vectorpainter.py "prompt='A Torii Gate.'" x=stroke target="./assets/Impression.jpg" result_path='./workspace/impression/torii_impression'
```

### Scream

```shell
CUDA_VISIBLE_DEVICES=3 python vectorpainter.py "prompt='A baby penguin.'" x=stroke target="./assets/scream.jpg" result_path='./workspace/scream/penguin_scream'
```

### Antimonocromatismo

```shell
CUDA_VISIBLE_DEVICES=4 python vectorpainter.py "prompt='A panda rowing a boat in a pond.'" x=stroke target="./assets/data_vis/sty/antimonocromatismo.png" result_path='./workspace/antimonocromatismo/panda_antimonocromatismo'
```
