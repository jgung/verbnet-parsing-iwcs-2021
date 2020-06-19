import glob
import os
import random
import subprocess


def output_results(_base_dir):
    max_scores = []
    for file in os.listdir(_base_dir):
        for summary in glob.glob(os.path.join(base_dir, file, '*summary*')):
            with open(summary) as lines:
                max_val = 0
                for line in list(lines)[1:]:
                    fields = line.split()
                    if not lines:
                        continue
                    max_val = max(float(fields[-1]), max_val)
                max_scores.append((summary, max_val))
    max_scores = sorted(max_scores, key=lambda x: x[1], reverse=True)
    for path, score in max_scores:
        print('%s\t%s' % (path, score))


if __name__ == '__main__':
    base_dir = "optim"
    SEED = 0

    base_config = "coling/config/semlink-pb-srl-transformer.json"
    train_file = "coling/datasets/semlink/train.txt"
    valid_file = "coling/datasets/semlink/dev.txt"

    override_vals = {
        "max_epochs": [16, 32],
        "batch_size": [64, 128, 196],
        "optimizer.lr.rate": [5e-4, 1e-3]
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

    r = random.Random(SEED)
    r.shuffle(overrides)
    with open("%s/runs.csv" % base_dir, mode='w') as out:
        for override in overrides:
            print(','.join(override))
            out.write(','.join(override))
            out.write('\n')

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

    output_results(base_dir)
