for ngradients in 32 64
do
	for nlosses in 32 64 128
		do
			python opt_main.py --ngradients ${ngradients} --nlosses ${nlosses} &> log_${ngradients}_${nlosses}.txt &
		done
done
