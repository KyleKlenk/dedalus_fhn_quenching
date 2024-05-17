#!/bin/zsh
N=${1:-4096}
L=${2:-2700}
basedir=./waves/
wavedir=$basedir/index_11
critdir=$basedir/index_10

dolin=0
dodns=1
stride=16
inflight=0
if [ ${dolin} = 1 ]; then	
	u0file=./waves/u0
fi

for ((gg = 0 ; gg <= 3 ; gg+=1)); do # 0, 3, 1
	Ufile=$wavedir/${gg}/U
	Pfile=$wavedir/${gg}/p
	ufile=$critdir/${gg}/U
	pfile=$critdir/${gg}/p
	w1file=$critdir/${gg}/W1
	w2file=$critdir/${gg}/W2
	v1file=$critdir/${gg}/V1
	v2file=$critdir/${gg}/V2
	
	for ((th = 0 ; th <= 0 ; th+=5)); do # originally: -75, 75, 5
		savedir=./dns/${gg}/${th}/
		mkdir -p $savedir
		# Run linear prediction of Us for family of Xs given Theta 
		if [ ${dolin} = 1 ]; then
			python3 linpred.py --savedir=$savedir --ufile=$ufile --v1file=$v1file --v2file=$v2file --w1file=$w1file --w2file=$w2file --Ufile=$Ufile --L=$L --N=8192 --u0file=$u0file --theta=$th
		fi
		# Run DNS prediction of Us for family of Xs given Theta 
		if [ ${dodns} = 1 ]; then
			inflight=$((${inflight}+1))
			python3 sweep.py --savedir=$savedir --Ufile=$Ufile --Pfile=$Pfile --ufile=$ufile --pfile=$pfile --theta=$th --N=${N} --dealias=1 &
		fi
		if [ ${inflight} = ${stride} ]; then
			wait
			inflight=0
		fi
	done
	#wait
done
