import random
import subprocess

if __name__ == '__main__':
    base_dir = "optim/"

    job_id = 1
    base_config = "data/config/experimental/srl-phrase-config.json"
    train_file = "data/datasets/conll05-srl/train-set.txt"
    valid_file = "data/datasets/conll05-srl/dev-set.txt"

    override_vals = {
        "max_epochs": [32],
        "optimizer.lr.rate": [0.0001, 0.00005],
        "optimizer.lr.params.max_lr": [0.001, 0.002],
        "optimizer.lr.params.step_size": [4]
    }

    overrides = []
    for key, val in override_vals.items():
        params = ['%s=%s' % (key, str(value)) for value in val]
        if not overrides:
            overrides = [[param] for param in params]
            continue
        new_overrides = []
        for override in overrides:
            for param in params:
                new_override = list(override)
                new_override.append(param)
                new_overrides.append(new_override)
        overrides = new_overrides

    random.shuffle(overrides)
    for override in overrides:
        print(','.join(override))

    for job_id, override in enumerate(overrides):
        params = [
            "python", "tfnlp/trainer.py",
            "--job-dir", "optim/%s" % job_id,
            "--config", base_config,
            "--resources", "data",
            "--train", train_file,
            "--valid", valid_file,
            "--param_overrides", ','.join(override)
        ]
        print(' '.join(params))
        subprocess.call(
            [
                "python", "tfnlp/trainer.py",
                "--job-dir", "optim/%s" % job_id,
                "--config", base_config,
                "--resources", "data",
                "--train", train_file,
                "--valid", valid_file,
                "--param_overrides", ','.join(override)
            ]
        )
        print(' '.join(params))
