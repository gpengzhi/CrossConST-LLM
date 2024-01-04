import argparse

from comet import load_from_checkpoint


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--file_src', required=True)
    parser.add_argument('--file_hyp', required=True)
    parser.add_argument('--file_ref', required=True)
    return parser


parser = get_parser()
args = parser.parse_args()

model = load_from_checkpoint(args.ckpt)


def compute_score(f_src, f_hyp, f_ref):
    lines_src = open(f_src, 'r').readlines()
    lines_hyp = open(f_hyp, 'r').readlines()
    lines_ref = open(f_ref, 'r').readlines()

    data = []

    for i in range(len(lines_src)):
        tmp = {}
        tmp["src"] = lines_src[i].rstrip('\n')
        tmp["mt"] = lines_hyp[i].rstrip('\n')
        tmp["ref"] = lines_ref[i].rstrip('\n')
        data.append(tmp)

    model_output = model.predict(data, batch_size=16, gpus=1)

    return model_output['system_score']


score = compute_score(args.file_src, args.file_hyp, args.file_ref)
print('COMET Score is {}'.format(round(score * 100, 2)))
