import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from common import ROOT, SEEDS, prepare_inputs, labelled, tune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=SEEDS)
    parser.add_argument('--output', type=Path, default=ROOT / 'trained')
    args = parser.parse_args()
    inputs = prepare_inputs(args.output / 'inputs')
    def call(script, *parameters):
        subprocess.run([sys.executable, str(ROOT / 'src' / script), *map(str, parameters)], check=True)
    for seed in args.seeds:
        call('teacher_train.py', '--variant', 'teacher', '--loss_mode', 'aspect_balanced', '--seed', seed,
             '--pair_dir', inputs, '--base_model_dir', args.base_model, '--out_root', args.output / 'teachers')
        teacher = args.output / f'teachers/teacher/seed_{seed:04d}/best_model'
        logits = args.output / f'teacher_logits/seed_{seed:04d}'
        call('teacher_logits.py', '--seed', seed, '--model_dir', teacher, '--out_dir', logits, '--pair_dir', inputs)
        call('student_train.py', '--variant', 'student', '--encoder_source', 'm2', '--seed', seed,
             '--data_dir', inputs, '--base_model_dir', args.base_model, '--m2_model_dir', teacher,
             '--codebook', ROOT / 'data/codebook.csv', '--teacher_dir', logits, '--out_root', args.output / 'students')
        student = args.output / f'students/student__encoder_m2/seed_{seed:04d}/best_model'
        from src.compact import export_compact
        final = args.output / f'compact_students/seed_{seed:04d}/best_model'
        export_compact(student, final)
        from predict import score_reviews
        dev = labelled('dev')
        dev_scores = score_reviews(final, [r['review_text_for_model'] for r in dev], batch_size=16, corpus_mode=False)
        labels = np.array([[r[k] for k in ['y_mention', 'y_positive_signal', 'y_negative_signal']] for r in dev]).transpose(0, 2, 1)
        threshold = tune(labels, dev_scores)
        np.save(final.parent / 'thresholds.npy', threshold)
        np.savez_compressed(final.parent / 'dev_scores.npz', scores=dev_scores, thresholds=threshold)


if __name__ == '__main__':
    main()
