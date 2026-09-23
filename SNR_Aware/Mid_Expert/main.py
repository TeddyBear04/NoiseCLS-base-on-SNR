"""Colab/molab-friendly entrypoint for the mid-SNR expert.

    python -u main.py teacher36 --config config/train_config.json
    python -u main.py bank36    --config config/train_config.json
"""

from tasks.run_36 import main


if __name__ == "__main__":
    main()
