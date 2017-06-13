#!/bin/bash

exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 

train=1 # train=1, eval=0

OPTIND=1
while getopts "e:g:t:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
	esac
done
shift $((OPTIND -1))

script='examples/drqa/train.py'
script=${script}' --log_file '$exp_dir'/exp'${exp}'.log'
script=${script}' -bs 32'
if [ $train -eq 1 ]; then
	script=${script}' --train_interval 3368'
fi

if [ $train -eq 0 ]; then
	script='examples/drqa/eval.py'
	script=${script}' --pretrained_model '${exp_dir}/exp${exp}' --datatype valid'
fi

case "$exp" in
	0) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --gpu $gpuid --tune_partial 1000 
		;;
	2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid
		;;
	3) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000
		;;
	4) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --fix_embeddings False
		;;
	5) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --rnn_padding True
		;;
	6) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --fix_embeddings False --rnn_padding True
		;;
	7) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --fix_embeddings False --rnn_padding True --optimizer adam --learning_rate 0.005
		;;
	8) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --rnn_padding True
		;;
	h11) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 ## r-net, _dep
		;;
	h11-1) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --hidden_size 75 ## r-net, _dep
		;;
	h12) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --hidden_size 75 ## r-net, GatedAttentionRNN
		;;
	h12-1) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --hidden_size 75 --rnn_padding True ## r-net, GatedAttentionRNN
		;;
	h12-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 -bs 16  ## r-net, GatedAttentionRNN
		;;
	h13) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --qp_bottleneck True ## r-net, GatedAttentionRNN
		;;
	h13-1) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --qp_bottleneck True --use_qemb False ## r-net, GatedAttentionRNN
		;;
	h13-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000  ## r-net, GatedAttentionRNN
		;;
	h13-3) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000  --use_qemb False ## r-net, GatedAttentionRNN
		;;
	h13-fix) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True ## r-net, GatedAttentionRNN
		;;
	h13-fix-bi) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True  ## r-net, GatedAttentionRNN
		;;
	h13-fix-bi-ldecay) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True # --train_interval 3368
		;;
	h13-fix-bi-ldecay-adam) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --learning_rate 0.0005 --optimizer adam # --train_interval 3368
		;;
	h13-fix-concat) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_concat True
		;;
	h13-fix-bi-concat) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --qp_concat True
		;;
	h13-fix-bi-concat-ldecay) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --qp_concat True --lrate_decay True
		;;
		######
		######
		###### PP matching
	h14-fix) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --pp_bottleneck True --hidden_size 75 -bs 24
		;;
	h14-fix-2) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --pp_bottleneck True --hidden_size 75 -bs 24
		;;
	h14-fix-bi) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --hidden_size 64 --rnn_type gru -bs 24
		;;
#	h14-fix-concat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_concat True --pp_concat True
#		;;
#	h14-fix-bi-concat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --qp_concat True --pp_birnn True --pp_concat True
#		;;
		######
		######
		###### Char + QP matching
	h15-fix-bi-ldecay-44) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --tune_partial 0 --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 -bs 20 
	;;
	h15-fix-bi-ldecay-44-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --tune_partial 0 --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 -bs 20 
	;;
	h15-fix-bi-ldecay-49) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.1  
	;;
	h15-fix-bi-ldecay-50) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.01
	;;



	debug) python $script -t squad --model_file $exp_dir/exp$exp --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000   ## r-net For debug
esac

