"""Entrypoint for the SSLAM-backed mid-SNR expert.

    python -u main.py teacher36 --config config/train_config.json
    python -u main.py student36 --config config/train_config.json --run run3_kd_crd
    python -u main.py test36    --config config/train_config.json --run run3_kd_crd
    python -u main.py report36  --config config/train_config.json

Needs `transformers<5` and `timm`.
"""

from tasks.run_36 import main


if __name__ == "__main__":
    main()
