# VectorPainter

## 🔥Quickstart

### Starry

```shell
CUDA_VISIBLE_DEVICES=7 python vectorpainter.py x=stroke prompt='A horse is drinking water by the lake. van Gogh style.' target="./assets/starry.jpg" result_path='./workspace/Starry/horse_starry' seed=100
CUDA_VISIBLE_DEVICES=2 python vectorpainter.py "prompt='A photo of Sydney opera house.'" x=stroke target="./assets/starry.jpg" result_path='./workspace/Starry/sydney_starry' seed=123123
CUDA_VISIBLE_DEVICES=0 python vectorpainter.py x=stroke prompt='A painting of a cat.' target="./assets/starry.jpg" result_path='./workspace/Starry/cat_starry' seed=100
CUDA_VISIBLE_DEVICES=6 python vectorpainter.py x=stroke "prompt='A mountain, with clouds in the sky.'" target="./assets/starry.jpg" result_path='./workspace/Starry/mountain_starry'
CUDA_VISIBLE_DEVICES=5 python vectorpainter.py x=stroke "prompt='A dragon-cat hybrid.'" target="./assets/starry.jpg" result_path='./workspace/Starry/dragon_cat_starry'

```

### Sunflowers

```shell 
CUDA_VISIBLE_DEVICES=2 python vectorpainter.py x=stroke "prompt='A bouquet of roses in a vase.'" target="./assets/sunflowers.jpg" result_path='./workspace/sunflowers/roses_sunflowers' seed=101
CUDA_VISIBLE_DEVICES=2 python vectorpainter.py x=stroke "prompt='A fire-breathing dragon.'" target="./assets/sunflowers.jpg" result_path='./workspace/sunflowers/dragon_sunflowers' seed=100
CUDA_VISIBLE_DEVICES=4 python vectorpainter.py "prompt='A photo of Sydney opera house.'" x=stroke target="./assets/sunflowers.jpg" result_path='./workspace/sunflowers/sydney_sunflowers'
CUDA_VISIBLE_DEVICES=1 python vectorpainter.py x=stroke "prompt='A bouquet of roses in a vase.'" target="./assets/sunflowers.jpg" x.pos_type='bez' x.pos_loss_weight=0.5 result_path='./workspace/sunflowers/roses_sunflowers_bez0.5' seed=100
```

### Field

```shell
CUDA_VISIBLE_DEVICES=7 python vectorpainter.py "prompt='A brightly colored mushroom growing on a log. van Gogh style.'" x=stroke target="./assets/Field.jpg" result_path='./workspace/Field/mushroom_Field'
CUDA_VISIBLE_DEVICES=4 python vectorpainter.py "prompt='A snail on a leaf. van Gogh style.'" x=stroke target="./assets/Field.jpg" result_path='./workspace/Field/snail_Field'
CUDA_VISIBLE_DEVICES=4 python vectorpainter.py "prompt='A photo of Sydney opera house on the sea.'" x=stroke target="./assets/Field.jpg" result_path='./workspace/Field/sydney_Field' seed=102
CUDA_VISIBLE_DEVICES=7 python vectorpainter.py "prompt='A bamboo ladder propped up against an oak tree.'" x=stroke target="./assets/Field.jpg" result_path='./workspace/Field/bamboo_Field' seed=100
```

### Impression

```shell
CUDA_VISIBLE_DEVICES=5 python vectorpainter.py "prompt='A Torii Gate.'" x=stroke target="./assets/Impression.jpg" result_path='./workspace/impression/torii_impression'
CUDA_VISIBLE_DEVICES=5 python vectorpainter.py "prompt='A beach with a cruise ship passing by.'" x=stroke target="./assets/Impression.jpg" result_path='./workspace/impression/beach_impression' seed=101
CUDA_VISIBLE_DEVICES=1 python vectorpainter.py "prompt='A panda rowing a boat in a pond.'" x=stroke target="./assets/Impression.jpg" result_path='./workspace/impression/panda_impression' seed=106
CUDA_VISIBLE_DEVICES=3 python vectorpainter.py "prompt='An underwater submarine.'" x=stroke target="./assets/Impression.jpg" result_path='./workspace/impression/submarine_impression'
```

### Scream

```shell
CUDA_VISIBLE_DEVICES=4 python vectorpainter.py "prompt='A baby penguin.'" x=stroke target="./assets/scream.jpg" result_path='./workspace/scream/penguin_scream'
```

### Antimonocromatismo

```shell
CUDA_VISIBLE_DEVICES=4 python vectorpainter.py "prompt='A panda rowing a boat in a pond.'" x=stroke target="./assets/data_vis/sty/antimonocromatismo.png" result_path='./workspace/antimonocromatismo/panda_antimonocromatismo'
```

### Majeur

```shell
CUDA_VISIBLE_DEVICES=6 python vectorpainter.py "prompt='The Great Pyramid.'" x=stroke target="./assets/majeur.jpg" result_path='./workspace/majeur/pyramid_majeur'
CUDA_VISIBLE_DEVICES=7 python vectorpainter.py "prompt='A Torii Gate.'" x=stroke target="./assets/majeur.jpg" result_path='./workspace/majeur/torii_majeur'
CUDA_VISIBLE_DEVICES=7 python vectorpainter.py "prompt='The Great Wall.'" x=stroke target="./assets/majeur.jpg" result_path='./workspace/majeur/wall_majeur'
CUDA_VISIBLE_DEVICES=3 python vectorpainter.py "prompt='The Eiffel Tower.'" x=stroke target="./assets/majeur.jpg" result_path='./workspace/majeur/eiffel_majeur' seed=101
```
