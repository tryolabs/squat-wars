<p align="center">
  <img alt="Squat Wars" src="https://user-images.githubusercontent.com/29715691/225422820-c305c46a-2432-4f2b-a504-3c94eb516637.png">
</p>

[Tryolabs](https://tryolabs.com/) demo featured at [Khipu 2023](https://khipu.ai/) consisting of a squat counter game running on a Raspberry Pi 4 together with a Coral TPU. The initial code was based on a [TensorFlow Lite pose estimation](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/raspberry_pi) example.

## Preview

https://github.com/tryolabs/squat-wars/assets/29715691/7fbc2d34-1c1b-4f8f-8e55-c76f73c70845

## How to Install

1. Clone the repository

   ```bash
   git clone git@github.com:tryolabs/squat-wars.git
   cd squat-wars
   ```

2. Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

   ```
   curl -sSL https://install.python-poetry.org | python -
   ```

3. Select Python `3.9.10` for your environment. If using pyenv, we suggest you do:
   ```bash
   pyenv shell 3.9.10
   pyenv which python | xargs poetry env use
   ```
4. Install dependencies and download models
   ```bash
   poetry install
   sh setup.sh
   ```

## How to Run

The game can be launched with the following command

```bash
poetry run python squat_wars/main.py
```

Additionally, there are a couple of flags that allow the behavior to be customized:

| Argument   | Description                      | Required | Default                                        |
| ---------- | -------------------------------- | -------- | ---------------------------------------------- |
| `--model`  | Path to the model `.tflite` file | No       | `squat_wars/models/movenet_thunder_tpu.tflite` |
| `--camera` | Camera ID. Set to 0 for webcam   | No       | `0`                                            |

> **Note** A camera needs to exist for the game to work.
> 
> It should also support a resolution of `800x448` as it is the current default. If it doesn't, the resolution can be changed [here](https://github.com/tryolabs/squat-wars/blob/da198e338d0ca76a53896805540866d3161b3ecb/squat_wars/game_state.py#LL9C1-L10C20).
> 
> Also, only single pose movenet models are supported at the moment. The name of the supported models are:
>
> - movenet_lightning
> - movenet_thunder
> - movenet_lightning_tpu
> - movenet_thunder_tpu

An example with both flags would be the following

```bash
poetry run python squat_wars/main.py --camera 0 --model squat_wars/models/movenet_lightning.tflite
```

## License

Copyright © 2022, [Tryolabs](https://tryolabs.com). Released under the [BSD 3-Clause](./LICENSE).
