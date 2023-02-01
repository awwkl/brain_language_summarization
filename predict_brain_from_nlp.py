import os
import argparse
import numpy as np

from utils.utils import run_class_time_CV_fmri_crossval_ridge
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    print(args)
        
    predict_feat_dict = {'nlp_feat_type':args.nlp_feat_type,
                         'nlp_feat_dir':args.nlp_feat_dir,
                         'layer':args.layer,
                         'seq_len':args.sequence_length}


    # loading fMRI data

    data = np.load('./data/fMRI/data_subject_{}.npy'.format(args.subject))          # (1211, ~27905)
    corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(data,
                                                                predict_feat_dict)

    fname = 'predict_{}_with_{}_layer_{}_len_{}'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length)
    fpath = os.path.join(args.output_dir, fname + '.npy')

    print(f'saving: {fpath}')
    np.save(fpath, {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t})

    
