"""Entrypoint for the SSLAM-backed SNR expert (band set by config["band_db"]).

    python -u main.py teacher36 --config config/train_config_high.json
    python -u main.py student36 --config config/train_config_high.json --run run3_kd_crd
    python -u main.py test36    --config config/train_config_high.json --run run3_kd_crd
    python -u main.py report36  --config config/train_config_high.json

Needs `transformers<5` and `timm`.
"""

from tasks.run_36 import main


if __name__ == "__main__":
    main()
