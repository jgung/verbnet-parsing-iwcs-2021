import random
import subprocess

if __name__ == '__main__':
    base_dir = "optim"

    base_config = "coling/config/semlink-pb-srl-albert-base.json"
    train_file = "coling/datasets/semlink/train.txt"
    valid_file = "coling/datasets/semlink/dev.txt"

    override_vals = {
        "max_epochs": [3, 4, 5],
        "batch_size": [16],
        "optimizer.lr.rate": [0.00003, 0.00004, 0.00005],
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
            "--job-dir", "%s/%s" % (base_dir, job_id),
            "--config", base_config,
            "--resources", "data",
            "--train", train_file,
            "--valid", valid_file,
            "--param_overrides", ','.join(override)
        ]
        print(' '.join(params))
        subprocess.call(params)
        print(' '.join(params))
